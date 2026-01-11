"""
Medusa Multi-Head Speculative Decoding
======================================

Medusa adds multiple "heads" to a single model to predict multiple future tokens
simultaneously, eliminating the need for a separate draft model.

Key innovation:
- Single model with additional lightweight prediction heads
- Each head predicts the token at position +1, +2, ..., +K
- Tree attention verifies multiple candidate sequences in parallel

Reference:
- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
  (Cai et al., 2024)
- NVIDIA TensorRT-LLM Medusa implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import time
import itertools


@dataclass
class MedusaConfig:
    """Configuration for Medusa decoding."""
    num_heads: int = 4  # Number of Medusa heads (predict +1, +2, ..., +num_heads)
    num_candidates_per_head: int = 5  # Top-k candidates from each head
    temperature: float = 1.0
    max_new_tokens: int = 128
    tree_width: int = 64  # Maximum number of candidate sequences to evaluate


class MedusaHead(nn.Module):
    """
    A single Medusa prediction head.

    Architecture: Hidden state -> [ResBlock] -> Linear -> Vocab logits

    The ResBlock helps the head learn residual predictions relative to
    the base model's predictions.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                )
            )

        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = hidden_states
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return self.head(x)


class MedusaModel(nn.Module):
    """
    Wraps a base LLM with Medusa heads for speculative decoding.

    The base model produces hidden states, and each Medusa head predicts
    the token at a different future position.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_heads: int = 4,
        num_layers_per_head: int = 1,
    ):
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config

        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size

        # Create Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, num_layers_per_head)
            for _ in range(num_heads)
        ])

        # Initialize heads with small weights
        self._init_heads()

    def _init_heads(self):
        """Initialize Medusa heads with small random weights."""
        for head in self.medusa_heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass through base model and all Medusa heads.

        Returns:
            Dictionary with:
            - base_logits: Logits from base model
            - medusa_logits: List of logits from each Medusa head
            - hidden_states: Final hidden states (for head training)
            - past_key_values: KV cache
        """
        # Get base model outputs with hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        # Get final hidden states
        hidden_states = outputs.hidden_states[-1]

        # Base model logits
        base_logits = outputs.logits

        # Get predictions from each Medusa head
        medusa_logits = [head(hidden_states) for head in self.medusa_heads]

        return {
            "base_logits": base_logits,
            "medusa_logits": medusa_logits,
            "hidden_states": hidden_states,
            "past_key_values": outputs.past_key_values,
        }


class MedusaDecoder:
    """
    Implements Medusa-style speculative decoding.

    Algorithm:
    1. Run base model to get hidden states
    2. Each Medusa head predicts top-k candidates for its position
    3. Form candidate sequences as Cartesian product of head predictions
    4. Use tree attention to verify all candidates in parallel
    5. Select the longest accepted sequence

    Example usage:
        >>> base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        >>> decoder = MedusaDecoder(base_model, tokenizer, num_heads=4)
        >>> output = decoder.generate("Once upon a time", max_new_tokens=50)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[MedusaConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained_heads: Optional[nn.ModuleList] = None,
    ):
        self.config = config or MedusaConfig()
        self.tokenizer = tokenizer
        self.device = device

        # Create Medusa model
        self.model = MedusaModel(
            base_model,
            num_heads=self.config.num_heads,
        ).to(device).eval()

        # Load pretrained heads if provided
        if pretrained_heads is not None:
            self.model.medusa_heads = pretrained_heads.to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_candidates(
        self,
        base_logits: torch.Tensor,
        medusa_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Generate candidate sequences from base and Medusa predictions.

        Takes top-k from base model and each Medusa head, then forms
        candidate sequences via Cartesian product.

        Args:
            base_logits: [batch, 1, vocab] - logits for position 0
            medusa_logits: List of [batch, 1, vocab] - logits for positions 1, 2, ...

        Returns:
            candidates: [batch, num_candidates, num_heads+1] - candidate token sequences
        """
        batch_size = base_logits.shape[0]
        k = self.config.num_candidates_per_head

        # Get top-k from base model (position 0)
        base_probs = F.softmax(base_logits[:, -1, :] / self.config.temperature, dim=-1)
        base_topk = torch.topk(base_probs, k=k, dim=-1)
        base_tokens = base_topk.indices  # [batch, k]

        # Get top-k from each Medusa head
        head_tokens = []
        for logits in medusa_logits:
            probs = F.softmax(logits[:, -1, :] / self.config.temperature, dim=-1)
            topk = torch.topk(probs, k=k, dim=-1)
            head_tokens.append(topk.indices)  # [batch, k]

        # Form candidate sequences (Cartesian product)
        # Limit total candidates to tree_width
        all_tokens = [base_tokens] + head_tokens

        # Generate all combinations
        num_positions = len(all_tokens)
        candidates = []

        # Use itertools product for generating combinations
        for indices in itertools.product(range(k), repeat=num_positions):
            if len(candidates) >= self.config.tree_width:
                break
            seq = torch.stack([all_tokens[i][0, idx] for i, idx in enumerate(indices)])
            candidates.append(seq)

        candidates = torch.stack(candidates).unsqueeze(0)  # [1, num_candidates, seq_len]
        return candidates

    @torch.no_grad()
    def _verify_candidates(
        self,
        prefix_ids: torch.Tensor,
        candidates: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify candidate sequences using tree attention.

        For each candidate sequence, check how many leading tokens
        the base model agrees with.

        Args:
            prefix_ids: Original input tokens
            candidates: [batch, num_candidates, seq_len] candidate sequences
            past_key_values: Cached KV states

        Returns:
            Tuple of (best_sequence, num_accepted)
        """
        batch_size, num_candidates, seq_len = candidates.shape

        # Simple verification: check each candidate independently
        # (A more efficient implementation would use tree attention)

        best_accepted = 0
        best_sequence = candidates[0, 0, :1]  # At minimum, keep first token

        for c in range(num_candidates):
            candidate = candidates[0, c, :]  # [seq_len]

            # Verify by running through base model
            test_input = torch.cat([prefix_ids, candidate.unsqueeze(0)], dim=1)

            outputs = self.model.base_model(
                input_ids=test_input,
                use_cache=False,
            )

            # Check agreement at each position
            logits = outputs.logits[0, prefix_ids.shape[1] - 1:-1, :]  # [seq_len, vocab]
            predicted = torch.argmax(logits, dim=-1)  # [seq_len]

            # Count matching positions
            matches = (predicted == candidate).int()
            num_accepted = 0
            for i in range(len(matches)):
                if matches[i] == 1:
                    num_accepted += 1
                else:
                    break

            if num_accepted > best_accepted:
                best_accepted = num_accepted
                best_sequence = candidate[:num_accepted + 1] if num_accepted < seq_len else candidate

        return best_sequence.unsqueeze(0), best_accepted

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_metrics: bool = True,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate text using Medusa speculative decoding.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            return_metrics: Whether to return performance metrics

        Returns:
            Tuple of (generated_text, metrics)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        metrics = {
            "total_iterations": 0,
            "total_tokens": 0,
            "avg_tokens_per_iteration": 0.0,
            "time_ms": 0.0,
        }

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        tokens_generated = 0
        num_iterations = 0
        start_time = time.time()

        while tokens_generated < max_new_tokens:
            # Get predictions from base model and Medusa heads
            outputs = self.model(
                input_ids=generated_ids,
                use_cache=False,  # Simplified: no KV cache
            )

            # Generate candidates
            candidates = self._get_candidates(
                outputs["base_logits"],
                outputs["medusa_logits"],
            )

            # Verify candidates
            accepted_tokens, num_accepted = self._verify_candidates(
                generated_ids,
                candidates,
            )

            # Append accepted tokens
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=1)
            tokens_generated += accepted_tokens.shape[1]
            num_iterations += 1

            # Check for EOS
            if self.tokenizer.eos_token_id in accepted_tokens:
                break

        elapsed_ms = (time.time() - start_time) * 1000

        metrics["total_iterations"] = num_iterations
        metrics["total_tokens"] = tokens_generated
        metrics["avg_tokens_per_iteration"] = tokens_generated / num_iterations if num_iterations > 0 else 0
        metrics["time_ms"] = elapsed_ms
        metrics["tokens_per_second"] = (tokens_generated * 1000) / elapsed_ms if elapsed_ms > 0 else 0

        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return (output_text, metrics) if return_metrics else (output_text, None)


class MedusaTrainer:
    """
    Trainer for Medusa heads.

    Training objective: Each head should predict the token at its offset position.
    Loss = sum of cross-entropy losses for each head.
    """

    def __init__(
        self,
        model: MedusaModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Only train Medusa heads, freeze base model
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        for param in self.model.medusa_heads.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.model.medusa_heads.parameters(),
            lr=learning_rate,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step for Medusa heads.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs [batch, seq_len]

        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(input_ids=input_ids, use_cache=False)

        num_heads = len(self.model.medusa_heads)
        total_loss = 0.0
        head_losses = {}

        # Compute loss for each head
        for i, logits in enumerate(outputs["medusa_logits"]):
            # Head i predicts token at position +i+1
            offset = i + 1

            if input_ids.shape[1] > offset:
                # Shift labels to align with predictions
                shifted_logits = logits[:, :-offset, :].contiguous()
                shifted_labels = labels[:, offset:].contiguous()

                loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss += loss
                head_losses[f"head_{i}_loss"] = loss.item()

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            **head_losses,
        }

    def save_heads(self, path: str):
        """Save trained Medusa heads."""
        torch.save(self.model.medusa_heads.state_dict(), path)

    def load_heads(self, path: str):
        """Load pretrained Medusa heads."""
        self.model.medusa_heads.load_state_dict(torch.load(path))
