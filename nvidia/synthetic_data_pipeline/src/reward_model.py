"""
Reward Model for Synthetic Data Evaluation
==========================================

Implements quality evaluation for synthetic data following NVIDIA Nemotron methodology.

Evaluation criteria (from Nemotron-4 340B Reward):
1. Helpfulness - Does the response address the instruction?
2. Correctness - Is the information accurate?
3. Coherence - Is the response well-organized?
4. Complexity - Does the response show appropriate depth?
5. Verbosity - Is the response appropriately detailed?

Reference:
- NVIDIA Nemotron-4 340B Reward Model
- "Nemotron-4 340B Technical Report"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


@dataclass
class QualityScores:
    """Scores for a single sample across quality dimensions."""
    helpfulness: float = 0.0
    correctness: float = 0.0
    coherence: float = 0.0
    complexity: float = 0.0
    verbosity: float = 0.0

    @property
    def overall(self) -> float:
        """Weighted average of all scores."""
        weights = {
            "helpfulness": 0.3,
            "correctness": 0.3,
            "coherence": 0.2,
            "complexity": 0.1,
            "verbosity": 0.1,
        }
        return sum(
            getattr(self, k) * v for k, v in weights.items()
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "helpfulness": self.helpfulness,
            "correctness": self.correctness,
            "coherence": self.coherence,
            "complexity": self.complexity,
            "verbosity": self.verbosity,
            "overall": self.overall,
        }


class RewardModelHead(nn.Module):
    """
    Reward model head for LLM-based evaluation.

    Takes hidden states and outputs reward scores.
    """

    def __init__(
        self,
        hidden_size: int,
        num_criteria: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_criteria = num_criteria

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for each criterion
        self.criterion_heads = nn.ModuleList([
            nn.Linear(hidden_size // 2, 1)
            for _ in range(num_criteria)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, hidden_size]

        Returns:
            scores: [batch, num_criteria]
        """
        features = self.mlp(hidden_states)

        scores = []
        for head in self.criterion_heads:
            score = head(features)
            scores.append(score)

        return torch.cat(scores, dim=-1)


class RewardModel:
    """
    Reward model for evaluating synthetic data quality.

    Can use either:
    1. A fine-tuned reward model (trained on preference data)
    2. LLM-as-judge with structured prompting

    Example usage:
        >>> reward_model = RewardModel(model, tokenizer, mode="llm_judge")
        >>> scores = reward_model.evaluate(instruction, response)
        >>> print(f"Overall score: {scores.overall:.2f}")
    """

    # Prompt for LLM-as-judge evaluation
    JUDGE_PROMPT = """Please evaluate the following response to an instruction on a scale of 1-5 for each criterion.

Instruction: {instruction}

Response: {response}

Evaluate on these criteria (1=Poor, 3=Adequate, 5=Excellent):

1. Helpfulness: Does the response address the instruction effectively?
2. Correctness: Is the information accurate and factual?
3. Coherence: Is the response well-organized and logical?
4. Complexity: Does the response show appropriate depth of thought?
5. Verbosity: Is the level of detail appropriate (not too brief or too long)?

Provide your scores in this exact format:
Helpfulness: [1-5]
Correctness: [1-5]
Coherence: [1-5]
Complexity: [1-5]
Verbosity: [1-5]

Scores:"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mode: str = "llm_judge",  # "llm_judge" or "reward_head"
        reward_head: Optional[RewardModelHead] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device

        if mode == "reward_head" and reward_head is not None:
            self.reward_head = reward_head.to(device).eval()
        else:
            self.reward_head = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def evaluate(
        self,
        instruction: str,
        response: str,
    ) -> QualityScores:
        """
        Evaluate a single instruction-response pair.

        Args:
            instruction: The instruction/prompt
            response: The generated response

        Returns:
            QualityScores object with scores for each criterion
        """
        if self.mode == "llm_judge":
            return self._evaluate_llm_judge(instruction, response)
        else:
            return self._evaluate_reward_head(instruction, response)

    def _evaluate_llm_judge(
        self,
        instruction: str,
        response: str,
    ) -> QualityScores:
        """Use LLM to evaluate quality."""
        prompt = self.JUDGE_PROMPT.format(
            instruction=instruction,
            response=response
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.1,  # Low temperature for consistent scoring
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Parse scores from generated text
        scores = self._parse_scores(generated)
        return scores

    def _parse_scores(self, text: str) -> QualityScores:
        """Parse scores from LLM judge output."""
        import re

        criteria_map = {
            "helpfulness": "helpfulness",
            "correctness": "correctness",
            "coherence": "coherence",
            "complexity": "complexity",
            "verbosity": "verbosity",
        }

        scores = QualityScores()

        for criterion, attr in criteria_map.items():
            # Look for patterns like "Helpfulness: 4" or "Helpfulness: 4/5"
            pattern = rf'{criterion}:\s*(\d)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                # Normalize to 0-1 range
                normalized_score = (score - 1) / 4.0
                setattr(scores, attr, normalized_score)
            else:
                # Default to neutral score if parsing fails
                setattr(scores, attr, 0.5)

        return scores

    def _evaluate_reward_head(
        self,
        instruction: str,
        response: str,
    ) -> QualityScores:
        """Use reward head for evaluation."""
        if self.reward_head is None:
            raise ValueError("Reward head not initialized")

        # Combine instruction and response
        text = f"Instruction: {instruction}\n\nResponse: {response}"
        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Get hidden states
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last token

        # Get reward scores
        raw_scores = self.reward_head(hidden_states)
        normalized_scores = torch.sigmoid(raw_scores)  # Normalize to 0-1

        scores_list = normalized_scores.squeeze().tolist()
        if not isinstance(scores_list, list):
            scores_list = [scores_list] * 5

        return QualityScores(
            helpfulness=scores_list[0],
            correctness=scores_list[1],
            coherence=scores_list[2],
            complexity=scores_list[3],
            verbosity=scores_list[4],
        )

    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[QualityScores]:
        """Evaluate multiple samples."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(samples, desc="Evaluating") if show_progress else samples

        for sample in iterator:
            scores = self.evaluate(
                sample["instruction"],
                sample["response"]
            )
            results.append(scores)

        return results


class QualityEvaluator:
    """
    Comprehensive quality evaluator combining multiple metrics.

    Includes:
    - LLM-based reward scoring
    - Rule-based heuristics
    - Diversity metrics
    """

    def __init__(
        self,
        reward_model: Optional[RewardModel] = None,
        min_response_length: int = 50,
        max_response_length: int = 2000,
    ):
        self.reward_model = reward_model
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length

    def evaluate(
        self,
        instruction: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a sample.

        Returns:
            Dictionary with all evaluation metrics
        """
        result = {
            "instruction": instruction,
            "response": response,
            "length_valid": True,
            "heuristic_scores": {},
            "reward_scores": None,
            "passed": True,
        }

        # Length checks
        response_len = len(response)
        if response_len < self.min_response_length:
            result["length_valid"] = False
            result["passed"] = False
            result["rejection_reason"] = "Response too short"
        elif response_len > self.max_response_length:
            result["length_valid"] = False
            result["passed"] = False
            result["rejection_reason"] = "Response too long"

        # Heuristic checks
        heuristics = self._compute_heuristics(instruction, response)
        result["heuristic_scores"] = heuristics

        if heuristics["repetition_ratio"] > 0.5:
            result["passed"] = False
            result["rejection_reason"] = "Too much repetition"

        if heuristics["instruction_overlap"] < 0.1:
            result["passed"] = False
            result["rejection_reason"] = "Response doesn't address instruction"

        # Reward model scores if available
        if self.reward_model and result["passed"]:
            reward_scores = self.reward_model.evaluate(instruction, response)
            result["reward_scores"] = reward_scores.to_dict()

            if reward_scores.overall < 0.3:
                result["passed"] = False
                result["rejection_reason"] = "Low quality score"

        return result

    def _compute_heuristics(
        self,
        instruction: str,
        response: str,
    ) -> Dict[str, float]:
        """Compute rule-based heuristic scores."""
        # Tokenize for analysis
        instruction_words = set(instruction.lower().split())
        response_words = response.lower().split()

        # Check for repetition (n-gram based)
        repetition_ratio = self._compute_repetition(response_words)

        # Check instruction-response overlap
        response_word_set = set(response_words)
        overlap = len(instruction_words & response_word_set) / len(instruction_words) if instruction_words else 0

        # Sentence count (rough coherence measure)
        sentence_count = response.count('.') + response.count('!') + response.count('?')

        # Average sentence length
        avg_sentence_len = len(response_words) / max(sentence_count, 1)

        return {
            "repetition_ratio": repetition_ratio,
            "instruction_overlap": overlap,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "word_count": len(response_words),
        }

    def _compute_repetition(self, words: List[str], n: int = 3) -> float:
        """Compute n-gram repetition ratio."""
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)

        return 1 - (len(unique_ngrams) / len(ngrams))

    def filter_samples(
        self,
        samples: List[Dict[str, str]],
        min_overall_score: float = 0.5,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples based on quality.

        Returns:
            Tuple of (passed_samples, rejected_samples)
        """
        passed = []
        rejected = []

        for sample in samples:
            result = self.evaluate(sample["instruction"], sample["response"])

            if result["passed"]:
                if result["reward_scores"] is None or result["reward_scores"]["overall"] >= min_overall_score:
                    sample["quality_scores"] = result
                    passed.append(sample)
                else:
                    result["rejection_reason"] = "Below minimum score threshold"
                    sample["quality_scores"] = result
                    rejected.append(sample)
            else:
                sample["quality_scores"] = result
                rejected.append(sample)

        return passed, rejected
