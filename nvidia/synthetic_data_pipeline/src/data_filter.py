"""
Data Filtering for Synthetic Data Quality
==========================================

Implements filtering strategies to ensure high-quality synthetic data:
- Quality-based filtering (using reward scores)
- Diversity filtering (remove near-duplicates)
- Safety filtering (remove harmful content)

Reference:
- NVIDIA NeMo Curator data filtering
- Nemotron-4 340B data curation pipeline
"""

import torch
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import re
import hashlib
from collections import defaultdict
import numpy as np


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    min_quality_score: float = 0.5
    max_repetition_ratio: float = 0.5
    min_instruction_length: int = 10
    max_instruction_length: int = 500
    min_response_length: int = 50
    max_response_length: int = 2000
    similarity_threshold: float = 0.85  # For deduplication
    blocked_phrases: List[str] = None

    def __post_init__(self):
        if self.blocked_phrases is None:
            self.blocked_phrases = [
                "as an ai",
                "i cannot",
                "i'm sorry",
                "i apologize",
            ]


class DataFilter:
    """
    Filters synthetic data based on quality criteria.

    Applies multiple filtering stages:
    1. Length filters
    2. Quality score filters
    3. Content filters (blocked phrases, patterns)
    4. Deduplication

    Example usage:
        >>> filter = DataFilter(config)
        >>> passed, rejected = filter.filter(samples)
        >>> print(f"Kept {len(passed)}/{len(samples)} samples")
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.stats = defaultdict(int)

    def filter(
        self,
        samples: List[Dict[str, str]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples based on all criteria.

        Args:
            samples: List of samples with 'instruction' and 'response' keys

        Returns:
            Tuple of (passed_samples, rejected_samples)
        """
        self.stats = defaultdict(int)
        passed = []
        rejected = []

        for sample in samples:
            is_valid, reason = self._check_sample(sample)

            if is_valid:
                passed.append(sample)
                self.stats["passed"] += 1
            else:
                sample["rejection_reason"] = reason
                rejected.append(sample)
                self.stats[f"rejected_{reason}"] += 1

        return passed, rejected

    def _check_sample(self, sample: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """Check if a sample passes all filters."""
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")

        # Length checks
        if len(instruction) < self.config.min_instruction_length:
            return False, "instruction_too_short"
        if len(instruction) > self.config.max_instruction_length:
            return False, "instruction_too_long"
        if len(response) < self.config.min_response_length:
            return False, "response_too_short"
        if len(response) > self.config.max_response_length:
            return False, "response_too_long"

        # Quality score check
        if "quality_scores" in sample:
            scores = sample["quality_scores"]
            if isinstance(scores, dict) and "reward_scores" in scores:
                if scores["reward_scores"] and scores["reward_scores"].get("overall", 1.0) < self.config.min_quality_score:
                    return False, "low_quality_score"

        # Blocked phrase check
        response_lower = response.lower()
        for phrase in self.config.blocked_phrases:
            if phrase in response_lower:
                return False, "blocked_phrase"

        # Repetition check
        if self._compute_repetition(response) > self.config.max_repetition_ratio:
            return False, "too_repetitive"

        # Empty or malformed check
        if not instruction.strip() or not response.strip():
            return False, "empty_content"

        return True, None

    def _compute_repetition(self, text: str, n: int = 3) -> float:
        """Compute n-gram repetition ratio."""
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0

        unique_ngrams = set(ngrams)
        return 1 - (len(unique_ngrams) / len(ngrams))

    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics."""
        return dict(self.stats)


class DiversityFilter:
    """
    Removes near-duplicate samples to ensure diversity.

    Uses multiple deduplication strategies:
    1. Exact match (hash-based)
    2. N-gram similarity
    3. Embedding-based similarity

    Reference:
    - NVIDIA NeMo Curator deduplication
    - MinHash for near-duplicate detection
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = False,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self.seen_hashes: Set[str] = set()

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except ImportError:
                print("sentence-transformers not installed. Falling back to n-gram similarity.")
                self.use_embeddings = False
                self.embedding_model = None
        else:
            self.embedding_model = None

    def filter(
        self,
        samples: List[Dict[str, str]],
        key: str = "instruction",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Remove near-duplicate samples.

        Args:
            samples: List of samples
            key: Key to use for deduplication ('instruction' or 'response')

        Returns:
            Tuple of (unique_samples, duplicate_samples)
        """
        if self.use_embeddings:
            return self._filter_with_embeddings(samples, key)
        else:
            return self._filter_with_ngrams(samples, key)

    def _filter_with_ngrams(
        self,
        samples: List[Dict[str, str]],
        key: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """N-gram based deduplication."""
        unique = []
        duplicates = []

        for sample in samples:
            text = sample.get(key, "")

            # Exact hash check
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.seen_hashes:
                duplicates.append(sample)
                continue

            # Check similarity with existing samples
            is_duplicate = False
            for existing in unique:
                existing_text = existing.get(key, "")
                similarity = self._ngram_similarity(text, existing_text)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicates.append(sample)
            else:
                unique.append(sample)
                self.seen_hashes.add(text_hash)

        return unique, duplicates

    def _ngram_similarity(
        self,
        text1: str,
        text2: str,
        n: int = 3,
    ) -> float:
        """Compute Jaccard similarity of n-grams."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if len(words1) < n or len(words2) < n:
            return 0.0

        ngrams1 = set(tuple(words1[i:i+n]) for i in range(len(words1) - n + 1))
        ngrams2 = set(tuple(words2[i:i+n]) for i in range(len(words2) - n + 1))

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union

    def _filter_with_embeddings(
        self,
        samples: List[Dict[str, str]],
        key: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Embedding-based deduplication."""
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [sample.get(key, "") for sample in samples]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)

        unique_indices = []
        duplicate_indices = []

        for i in range(len(samples)):
            is_duplicate = False
            for j in unique_indices:
                if similarities[i, j] > self.similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicate_indices.append(i)
            else:
                unique_indices.append(i)

        unique = [samples[i] for i in unique_indices]
        duplicates = [samples[i] for i in duplicate_indices]

        return unique, duplicates

    def reset(self):
        """Reset seen hashes for new batch."""
        self.seen_hashes = set()


class SafetyFilter:
    """
    Filters out potentially harmful or unsafe content.

    Checks for:
    - Explicit content
    - Violence
    - Hate speech
    - Personal information
    - Dangerous instructions
    """

    # Patterns that indicate potentially unsafe content
    UNSAFE_PATTERNS = [
        r'\b(kill|murder|attack|bomb|weapon)\b',
        r'\b(hack|exploit|vulnerability|crack)\b',
        r'\b(drug|narcotic|cocaine|heroin)\b',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # SSN pattern
    ]

    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = patterns or self.UNSAFE_PATTERNS
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def is_safe(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is safe.

        Returns:
            Tuple of (is_safe, matched_pattern_or_None)
        """
        for pattern, compiled in zip(self.patterns, self.compiled_patterns):
            if compiled.search(text):
                return False, pattern
        return True, None

    def filter(
        self,
        samples: List[Dict[str, str]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter out unsafe samples."""
        safe = []
        unsafe = []

        for sample in samples:
            text = f"{sample.get('instruction', '')} {sample.get('response', '')}"
            is_safe, pattern = self.is_safe(text)

            if is_safe:
                safe.append(sample)
            else:
                sample["unsafe_pattern"] = pattern
                unsafe.append(sample)

        return safe, unsafe


class CombinedFilter:
    """
    Combines multiple filters into a single pipeline.
    """

    def __init__(
        self,
        quality_filter: Optional[DataFilter] = None,
        diversity_filter: Optional[DiversityFilter] = None,
        safety_filter: Optional[SafetyFilter] = None,
    ):
        self.quality_filter = quality_filter or DataFilter()
        self.diversity_filter = diversity_filter or DiversityFilter()
        self.safety_filter = safety_filter or SafetyFilter()

    def filter(
        self,
        samples: List[Dict[str, str]],
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Apply all filters in sequence.

        Returns:
            Tuple of (passed_samples, rejected_by_filter)
        """
        rejected = {
            "quality": [],
            "diversity": [],
            "safety": [],
        }

        # Stage 1: Quality filter
        samples, rejected["quality"] = self.quality_filter.filter(samples)

        # Stage 2: Safety filter
        samples, rejected["safety"] = self.safety_filter.filter(samples)

        # Stage 3: Diversity filter
        samples, rejected["diversity"] = self.diversity_filter.filter(samples)

        return samples, rejected

    def get_summary(self, rejected: Dict[str, List]) -> Dict[str, Any]:
        """Get summary of filtering results."""
        return {
            "rejected_by_quality": len(rejected["quality"]),
            "rejected_by_safety": len(rejected["safety"]),
            "rejected_by_diversity": len(rejected["diversity"]),
            "total_rejected": sum(len(v) for v in rejected.values()),
        }
