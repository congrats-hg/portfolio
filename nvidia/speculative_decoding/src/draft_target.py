"""
Draft-Target Speculative Decoding Implementation
=================================================

This module implements the classic speculative decoding approach where:
1. A small "draft" model generates K candidate tokens quickly
2. A large "target" model verifies all K tokens in a single forward pass
3. Accepted tokens are kept, rejected tokens trigger resampling

Key insight: Verification of K tokens costs ~same as generating 1 token
(due to parallelism), but we accept multiple tokens on average.

Reference:
- "Accelerating Large Language Model Decoding with Speculative Sampling"
  (Leviathan et al., 2023)
- NVIDIA TensorRT-LLM speculative decoding implementation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import time


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: int = 5  # K: number of draft tokens to generate
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    use_cache: bool = True


@dataclass
class DecodingMetrics:
    """Metrics collected during decoding."""
    total_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return (self.accepted_tokens * 1000) / self.total_time_ms

    @property
    def speedup_factor(self) -> float:
        """Estimated speedup vs autoregressive decoding."""
        if self.accepted_tokens == 0:
            return 1.0
        # Speedup ≈ accepted_tokens / num_forward_passes
        # In ideal case with K speculative tokens and 100% acceptance: K+1
        return self.acceptance_rate * 5 + 1  # Approximate


class DraftTargetDecoder:
    """
    Implements Draft-Target Speculative Decoding.

    This technique uses a smaller "draft" model to generate candidate tokens
    and a larger "target" model to verify them in parallel.

    Algorithm:
    1. Draft model generates K tokens autoregressively
    2. Target model processes prefix + K draft tokens in one forward pass
    3. Compare distributions at each position:
       - Accept token if target agrees (or sample from residual distribution)
       - Reject token and resample from target distribution
    4. Keep accepted tokens, repeat from last accepted position

    Example usage:
        >>> draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> target_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> decoder = DraftTargetDecoder(draft_model, target_model, tokenizer)
        >>> output = decoder.generate("Once upon a time", max_new_tokens=50)
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[SpeculativeDecodingConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.draft_model = draft_model.to(device).eval()
        self.target_model = target_model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config or SpeculativeDecodingConfig()
        self.device = device

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sample_from_distribution(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a token from logits with temperature, top-p, and top-k.

        Returns:
            Tuple of (sampled_token, probability)
        """
        if temperature == 0:
            # Greedy decoding
            probs = F.softmax(logits, dim=-1)
            token = torch.argmax(logits, dim=-1)
            return token, probs[..., token]

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token_prob = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)

        return token, token_prob

    @torch.no_grad()
    def _generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        num_tokens: int = 5,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[Tuple]]:
        """
        Generate K draft tokens using the draft model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            past_key_values: Cached key-values from previous generation
            num_tokens: Number of draft tokens to generate

        Returns:
            Tuple of (draft_tokens, draft_probs, new_past_key_values)
        """
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids

        for _ in range(num_tokens):
            outputs = self.draft_model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=self.config.use_cache,
            )

            # Get logits for the last position
            logits = outputs.logits[:, -1, :]

            # Sample token
            token, prob = self._sample_from_distribution(
                logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            draft_tokens.append(token)
            draft_probs.append(F.softmax(logits, dim=-1))

            # Update for next iteration
            current_ids = token.unsqueeze(-1)
            past_key_values = outputs.past_key_values

        return torch.stack(draft_tokens, dim=1), draft_probs, past_key_values

    @torch.no_grad()
    def _verify_draft_tokens(
        self,
        prefix_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: List[torch.Tensor],
        target_past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, int, Optional[Tuple]]:
        """
        Verify draft tokens using the target model.

        Implements the speculative sampling algorithm:
        - For each position, compare draft and target distributions
        - Accept with probability min(1, p_target / p_draft)
        - On rejection, sample from adjusted distribution

        Args:
            prefix_ids: Original input tokens
            draft_tokens: K tokens generated by draft model
            draft_probs: Probability distributions from draft model
            target_past_key_values: Cached KV from target model

        Returns:
            Tuple of (accepted_tokens, num_accepted, new_past_key_values)
        """
        batch_size = prefix_ids.shape[0]
        num_draft = draft_tokens.shape[1]

        # Concatenate prefix with draft tokens for single forward pass
        full_ids = torch.cat([prefix_ids, draft_tokens], dim=1)

        # Get target model predictions for all positions
        outputs = self.target_model(
            input_ids=full_ids if target_past_key_values is None else draft_tokens,
            past_key_values=target_past_key_values,
            use_cache=self.config.use_cache,
        )

        # Extract logits for draft token positions
        # Shape: [batch, num_draft + 1, vocab]
        if target_past_key_values is None:
            target_logits = outputs.logits[:, -(num_draft + 1):, :]
        else:
            target_logits = outputs.logits

        target_probs = F.softmax(target_logits / self.config.temperature, dim=-1)

        # Verify each draft token
        accepted_tokens = []
        num_accepted = 0

        for i in range(num_draft):
            draft_token = draft_tokens[:, i]
            p_draft = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            p_target = target_probs[:, i, :].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)

            # Acceptance probability: min(1, p_target / p_draft)
            accept_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)

            # Sample acceptance decision
            accept = torch.rand(batch_size, device=self.device) < accept_prob

            if accept.all():
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                # Rejection: sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft)
                adjusted_probs = torch.clamp(
                    target_probs[:, i, :] - draft_probs[i],
                    min=0
                )
                adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)

                # Handle case where adjustment leads to zero distribution
                if adjusted_probs.sum() < 1e-6:
                    adjusted_probs = target_probs[:, i, :]

                new_token = torch.multinomial(adjusted_probs, num_samples=1).squeeze(-1)
                accepted_tokens.append(new_token)
                num_accepted += 1
                break

        # If all draft tokens accepted, sample one more from target
        if num_accepted == num_draft:
            bonus_token, _ = self._sample_from_distribution(
                target_logits[:, -1, :] * self.config.temperature,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            accepted_tokens.append(bonus_token)
            num_accepted += 1

        accepted = torch.stack(accepted_tokens, dim=1) if accepted_tokens else torch.tensor([], device=self.device)

        return accepted, num_accepted, outputs.past_key_values

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_metrics: bool = True,
    ) -> Tuple[str, Optional[DecodingMetrics]]:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            return_metrics: Whether to return decoding metrics

        Returns:
            Tuple of (generated_text, metrics)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        metrics = DecodingMetrics()

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        draft_past_kv = None
        target_past_kv = None

        start_time = time.time()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Phase 1: Generate draft tokens
            draft_start = time.time()
            draft_tokens, draft_probs, draft_past_kv = self._generate_draft_tokens(
                generated_ids if draft_past_kv is None else generated_ids[:, -1:],
                past_key_values=draft_past_kv,
                num_tokens=min(self.config.num_speculative_tokens, max_new_tokens - tokens_generated),
            )
            metrics.draft_time_ms += (time.time() - draft_start) * 1000

            # Phase 2: Verify draft tokens
            verify_start = time.time()
            accepted, num_accepted, target_past_kv = self._verify_draft_tokens(
                generated_ids,
                draft_tokens,
                draft_probs,
                target_past_key_values=target_past_kv,
            )
            metrics.verify_time_ms += (time.time() - verify_start) * 1000

            # Update metrics
            metrics.total_tokens += draft_tokens.shape[1]
            metrics.accepted_tokens += num_accepted
            metrics.rejected_tokens += draft_tokens.shape[1] - num_accepted + (1 if num_accepted == draft_tokens.shape[1] + 1 else 0)

            # Append accepted tokens
            if accepted.numel() > 0:
                generated_ids = torch.cat([generated_ids, accepted], dim=1)
                tokens_generated += num_accepted

            # Clear KV cache if rejection occurred (simplified approach)
            if num_accepted < draft_tokens.shape[1]:
                draft_past_kv = None
                target_past_kv = None

            # Check for EOS token
            if self.tokenizer.eos_token_id in accepted:
                break

        metrics.total_time_ms = (time.time() - start_time) * 1000

        # Decode output
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return (output_text, metrics) if return_metrics else (output_text, None)

    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Run benchmark on multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens per generation

        Returns:
            Dictionary with aggregate metrics
        """
        all_metrics = []

        for prompt in prompts:
            _, metrics = self.generate(prompt, max_new_tokens=max_new_tokens)
            all_metrics.append(metrics)

        # Aggregate metrics
        total_accepted = sum(m.accepted_tokens for m in all_metrics)
        total_tokens = sum(m.total_tokens for m in all_metrics)
        total_time = sum(m.total_time_ms for m in all_metrics)

        return {
            "num_prompts": len(prompts),
            "avg_acceptance_rate": total_accepted / total_tokens if total_tokens > 0 else 0,
            "avg_tokens_per_second": (total_accepted * 1000) / total_time if total_time > 0 else 0,
            "total_tokens_generated": total_accepted,
            "total_time_ms": total_time,
            "individual_metrics": [
                {
                    "acceptance_rate": m.acceptance_rate,
                    "tokens_per_second": m.tokens_per_second,
                    "speedup_factor": m.speedup_factor,
                }
                for m in all_metrics
            ],
        }


class AutoregressiveDecoder:
    """
    Standard autoregressive decoder for baseline comparison.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Tuple[str, float]:
        """Generate text and return (text, time_ms)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        start_time = time.time()

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text, elapsed_ms
