"""
Diversity Sampling for Synthetic Data
=====================================

Ensures diverse coverage across topics, difficulty levels, and styles.

Techniques:
1. Embedding-based clustering
2. Topic modeling
3. Stratified sampling

Reference:
- NVIDIA Nemotron data diversity strategies
- "Diversity is All You Need" for instruction tuning
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random


@dataclass
class DiversitySamplerConfig:
    """Configuration for diversity sampling."""
    num_clusters: int = 10
    samples_per_cluster: int = 100
    min_cluster_size: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_seed: int = 42


class EmbeddingClusterer:
    """
    Clusters samples based on embedding similarity.

    Uses K-means clustering on sentence embeddings to group
    similar samples together.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_clusters: int = 10,
    ):
        self.num_clusters = num_clusters

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("sentence-transformers required. Install with: pip install sentence-transformers")
            self.model = None

    def cluster(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster texts using K-means.

        Args:
            texts: List of texts to cluster

        Returns:
            Tuple of (cluster_labels, embeddings)
        """
        if self.model is None:
            # Fallback: random clustering
            labels = np.random.randint(0, self.num_clusters, size=len(texts))
            embeddings = np.random.randn(len(texts), 384)
            return labels, embeddings

        from sklearn.cluster import KMeans

        # Get embeddings
        print("Computing embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Cluster
        print("Clustering...")
        kmeans = KMeans(
            n_clusters=min(self.num_clusters, len(texts)),
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)

        return labels, embeddings

    def get_cluster_centers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Get center embedding for each cluster."""
        centers = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            centers[cluster_id] = embeddings[mask].mean(axis=0)
        return centers


class TopicExtractor:
    """
    Extracts topics from samples for stratified sampling.

    Uses simple keyword-based topic assignment or can integrate
    with topic modeling libraries.
    """

    # Default topic keywords
    TOPIC_KEYWORDS = {
        "coding": ["code", "program", "function", "algorithm", "debug", "python", "java"],
        "math": ["calculate", "math", "equation", "solve", "number", "formula"],
        "writing": ["write", "essay", "story", "article", "blog", "creative"],
        "science": ["science", "experiment", "research", "hypothesis", "physics", "chemistry"],
        "business": ["business", "market", "strategy", "company", "profit", "sales"],
        "general": [],  # Fallback
    }

    def __init__(self, topic_keywords: Optional[Dict[str, List[str]]] = None):
        self.topic_keywords = topic_keywords or self.TOPIC_KEYWORDS

    def extract_topic(self, text: str) -> str:
        """Extract primary topic from text."""
        text_lower = text.lower()

        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            if topic == "general":
                continue
            score = sum(1 for kw in keywords if kw in text_lower)
            topic_scores[topic] = score

        if not topic_scores or max(topic_scores.values()) == 0:
            return "general"

        return max(topic_scores, key=topic_scores.get)

    def extract_topics_batch(self, texts: List[str]) -> List[str]:
        """Extract topics for multiple texts."""
        return [self.extract_topic(text) for text in texts]


class DiversitySampler:
    """
    Samples diverse subset from synthetic data.

    Ensures:
    1. Coverage across different topics/clusters
    2. Balance between easy and complex samples
    3. Variety in response styles

    Example usage:
        >>> sampler = DiversitySampler(config)
        >>> diverse_samples = sampler.sample(all_samples, num_samples=1000)
    """

    def __init__(self, config: Optional[DiversitySamplerConfig] = None):
        self.config = config or DiversitySamplerConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.clusterer = EmbeddingClusterer(
            model_name=self.config.embedding_model,
            num_clusters=self.config.num_clusters
        )
        self.topic_extractor = TopicExtractor()

    def sample(
        self,
        samples: List[Dict[str, str]],
        num_samples: int,
        strategy: str = "cluster_balanced",  # "cluster_balanced", "topic_balanced", "random"
    ) -> List[Dict[str, str]]:
        """
        Sample diverse subset from samples.

        Args:
            samples: Full list of samples
            num_samples: Target number of samples
            strategy: Sampling strategy

        Returns:
            Diverse subset of samples
        """
        if len(samples) <= num_samples:
            return samples

        if strategy == "cluster_balanced":
            return self._sample_cluster_balanced(samples, num_samples)
        elif strategy == "topic_balanced":
            return self._sample_topic_balanced(samples, num_samples)
        else:
            return random.sample(samples, num_samples)

    def _sample_cluster_balanced(
        self,
        samples: List[Dict[str, str]],
        num_samples: int,
    ) -> List[Dict[str, str]]:
        """Sample equally from each cluster."""
        # Get texts for clustering
        texts = [s.get("instruction", "") for s in samples]

        # Cluster
        labels, embeddings = self.clusterer.cluster(texts)

        # Group samples by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append((i, samples[i]))

        # Calculate samples per cluster
        num_clusters = len(clusters)
        base_per_cluster = num_samples // num_clusters
        extra = num_samples % num_clusters

        # Sample from each cluster
        sampled = []
        for cluster_id, cluster_samples in clusters.items():
            n = base_per_cluster + (1 if cluster_id < extra else 0)
            n = min(n, len(cluster_samples))

            # Sample diverse within cluster
            if len(cluster_samples) <= n:
                sampled.extend([s for _, s in cluster_samples])
            else:
                # Sample based on distance from cluster center
                indices = [i for i, _ in cluster_samples]
                cluster_embeddings = embeddings[indices]
                center = cluster_embeddings.mean(axis=0)

                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                # Mix of close and far from center
                sorted_indices = np.argsort(distances)
                selected = list(sorted_indices[:n//2]) + list(sorted_indices[-(n - n//2):])
                selected = selected[:n]

                for idx in selected:
                    sampled.append(cluster_samples[idx][1])

        return sampled

    def _sample_topic_balanced(
        self,
        samples: List[Dict[str, str]],
        num_samples: int,
    ) -> List[Dict[str, str]]:
        """Sample equally from each topic."""
        # Extract topics
        texts = [s.get("instruction", "") for s in samples]
        topics = self.topic_extractor.extract_topics_batch(texts)

        # Group by topic
        topic_groups = defaultdict(list)
        for i, topic in enumerate(topics):
            topic_groups[topic].append(samples[i])

        # Calculate samples per topic
        num_topics = len(topic_groups)
        base_per_topic = num_samples // num_topics
        extra = num_samples % num_topics

        # Sample from each topic
        sampled = []
        for i, (topic, group) in enumerate(topic_groups.items()):
            n = base_per_topic + (1 if i < extra else 0)
            n = min(n, len(group))

            if len(group) <= n:
                sampled.extend(group)
            else:
                sampled.extend(random.sample(group, n))

        return sampled

    def compute_diversity_score(
        self,
        samples: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for a set of samples.

        Returns:
            Dictionary with diversity metrics
        """
        texts = [s.get("instruction", "") for s in samples]

        # Topic diversity
        topics = self.topic_extractor.extract_topics_batch(texts)
        topic_counts = defaultdict(int)
        for topic in topics:
            topic_counts[topic] += 1

        topic_entropy = self._compute_entropy(list(topic_counts.values()))

        # Length diversity
        lengths = [len(s.get("response", "")) for s in samples]
        length_std = np.std(lengths)

        # Vocabulary diversity
        all_words = set()
        for s in samples:
            all_words.update(s.get("response", "").lower().split())
        vocab_size = len(all_words)

        return {
            "topic_entropy": topic_entropy,
            "topic_distribution": dict(topic_counts),
            "length_std": length_std,
            "vocab_size": vocab_size,
            "avg_length": np.mean(lengths),
        }

    def _compute_entropy(self, counts: List[int]) -> float:
        """Compute entropy of distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0

        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)


class StratifiedSampler:
    """
    Stratified sampling based on multiple attributes.
    """

    def __init__(self, attributes: List[str] = None):
        """
        Args:
            attributes: Sample attributes to stratify by
        """
        self.attributes = attributes or ["topic", "complexity", "length_bucket"]

    def _get_strata_key(self, sample: Dict) -> str:
        """Get stratification key for sample."""
        values = []
        for attr in self.attributes:
            if attr == "length_bucket":
                length = len(sample.get("response", ""))
                if length < 100:
                    values.append("short")
                elif length < 500:
                    values.append("medium")
                else:
                    values.append("long")
            elif attr in sample:
                values.append(str(sample[attr]))
            else:
                values.append("unknown")
        return "_".join(values)

    def sample(
        self,
        samples: List[Dict[str, str]],
        num_samples: int,
        min_per_stratum: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Sample with stratification.

        Args:
            samples: All samples
            num_samples: Target count
            min_per_stratum: Minimum samples per stratum

        Returns:
            Stratified sample
        """
        # Group by strata
        strata = defaultdict(list)
        for sample in samples:
            key = self._get_strata_key(sample)
            strata[key].append(sample)

        # Calculate samples per stratum
        num_strata = len(strata)
        if num_strata == 0:
            return []

        # Proportional allocation
        total = len(samples)
        sampled = []

        for key, group in strata.items():
            proportion = len(group) / total
            n = max(min_per_stratum, int(num_samples * proportion))
            n = min(n, len(group))

            sampled.extend(random.sample(group, n))

        # Trim if over
        if len(sampled) > num_samples:
            sampled = random.sample(sampled, num_samples)

        return sampled
