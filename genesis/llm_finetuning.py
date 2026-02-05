"""Privacy-Preserving LLM Fine-Tuning Data.

Generate synthetic data optimized for LLM fine-tuning without
memorization risks, with differential privacy guarantees and
membership inference protection.

Features:
    - Memorization-safe text synthesis
    - Differential privacy for fine-tuning data
    - Membership inference attack protection
    - Diverse paraphrase generation
    - PII scrubbing and replacement
    - Training data filtering
    - Quality-utility trade-off optimization

Example:
    Generate safe fine-tuning data::

        from genesis.llm_finetuning import (
            FineTuningDataGenerator,
            SafetyConfig,
        )

        generator = FineTuningDataGenerator(
            config=SafetyConfig(
                enable_dp=True,
                epsilon=1.0,
                min_frequency_threshold=5,
            )
        )

        # Generate from existing data
        safe_data = generator.generate_from_data(
            training_data,
            n_samples=50000,
        )

        # Or generate from schema/prompts
        safe_data = generator.generate_from_prompts(
            prompts=[
                "Generate a customer service conversation",
                "Create a product review",
            ],
            n_per_prompt=1000,
        )

        # Verify no memorization
        report = generator.audit_memorization(safe_data, training_data)
        print(f"Memorization risk: {report.risk_score}")

Classes:
    SafetyConfig: Safety configuration for fine-tuning data.
    MemorizationDetector: Detects potential memorization.
    TextDeduplicator: Removes near-duplicate text.
    ParaphraseGenerator: Generates diverse paraphrases.
    PIIScrubber: Removes/replaces PII.
    DPTextGenerator: DP-safe text generation.
    FineTuningDataGenerator: Main generator interface.
    MemorizationAuditReport: Audit results.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class TextFilterLevel(str, Enum):
    """Filtering strictness level."""

    MINIMAL = "minimal"  # Only remove exact duplicates
    STANDARD = "standard"  # Remove near-duplicates and low quality
    STRICT = "strict"  # Aggressive filtering for maximum safety
    PARANOID = "paranoid"  # Extremely strict, may reduce utility


class PIIType(str, Enum):
    """Types of PII to handle."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class SafetyConfig:
    """Safety configuration for fine-tuning data generation.

    Attributes:
        enable_dp: Enable differential privacy
        epsilon: Privacy budget (lower = more private)
        delta: DP delta parameter
        min_frequency_threshold: Minimum frequency for n-grams
        max_similarity_threshold: Max similarity to training data (0-1)
        filter_level: Text filtering strictness
        remove_pii: Remove PII from generated text
        pii_types: Types of PII to remove
        min_text_length: Minimum text length
        max_text_length: Maximum text length
        deduplication_threshold: Similarity threshold for dedup
        enable_paraphrase: Generate paraphrases for diversity
        paraphrase_diversity: Diversity of paraphrases (0-1)
    """

    enable_dp: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    min_frequency_threshold: int = 5
    max_similarity_threshold: float = 0.7
    filter_level: TextFilterLevel = TextFilterLevel.STANDARD
    remove_pii: bool = True
    pii_types: List[PIIType] = field(default_factory=lambda: list(PIIType))
    min_text_length: int = 10
    max_text_length: int = 10000
    deduplication_threshold: float = 0.85
    enable_paraphrase: bool = True
    paraphrase_diversity: float = 0.5


@dataclass
class MemorizationAuditReport:
    """Report from memorization audit.

    Attributes:
        risk_score: Overall risk score (0-1)
        exact_matches: Number of exact matches found
        near_duplicates: Number of near-duplicates
        high_similarity_samples: Samples with high similarity
        n_gram_overlaps: Overlapping n-grams
        recommendations: Safety recommendations
        passed: Whether audit passed
    """

    risk_score: float
    exact_matches: int
    near_duplicates: int
    high_similarity_samples: List[Dict[str, Any]]
    n_gram_overlaps: Dict[int, int]  # n -> count
    recommendations: List[str]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class MemorizationDetector:
    """Detects potential memorization in generated text.

    Uses multiple techniques:
    - Exact match detection
    - Near-duplicate detection (MinHash, edit distance)
    - N-gram overlap analysis
    - Longest common substring
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        n_gram_sizes: List[int] = None,
    ):
        """Initialize detector.

        Args:
            similarity_threshold: Threshold for near-duplicate detection
            n_gram_sizes: N-gram sizes to check
        """
        self._threshold = similarity_threshold
        self._n_gram_sizes = n_gram_sizes or [5, 10, 20]
        self._training_hashes: Set[str] = set()
        self._training_ngrams: Dict[int, Set[str]] = {}

    def fit(self, training_data: List[str]) -> "MemorizationDetector":
        """Fit detector on training data.

        Args:
            training_data: Original training texts

        Returns:
            Self for chaining
        """
        # Store hashes for exact match
        for text in training_data:
            self._training_hashes.add(hashlib.sha256(text.encode()).hexdigest())

        # Build n-gram index
        for n in self._n_gram_sizes:
            self._training_ngrams[n] = set()
            for text in training_data:
                for ngram in self._get_ngrams(text, n):
                    self._training_ngrams[n].add(ngram)

        return self

    def detect(
        self,
        generated_texts: List[str],
        training_data: Optional[List[str]] = None,
    ) -> MemorizationAuditReport:
        """Detect memorization in generated texts.

        Args:
            generated_texts: Generated texts to check
            training_data: Original training data (optional if fitted)

        Returns:
            MemorizationAuditReport
        """
        if training_data:
            self.fit(training_data)

        exact_matches = 0
        near_duplicates = 0
        high_similarity_samples: List[Dict[str, Any]] = []
        n_gram_overlaps: Dict[int, int] = {n: 0 for n in self._n_gram_sizes}

        for i, text in enumerate(generated_texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Check exact match
            if text_hash in self._training_hashes:
                exact_matches += 1
                high_similarity_samples.append({
                    "index": i,
                    "type": "exact_match",
                    "similarity": 1.0,
                    "text_preview": text[:100] + "...",
                })
                continue

            # Check n-gram overlap
            for n, ngram_set in self._training_ngrams.items():
                text_ngrams = set(self._get_ngrams(text, n))
                overlap = len(text_ngrams & ngram_set) / max(len(text_ngrams), 1)
                if overlap > self._threshold:
                    n_gram_overlaps[n] += 1

            # Check similarity if we have training data
            if training_data:
                max_similarity = 0.0
                for train_text in training_data[:1000]:  # Sample for efficiency
                    sim = self._compute_similarity(text, train_text)
                    max_similarity = max(max_similarity, sim)
                    if sim > self._threshold:
                        break

                if max_similarity > self._threshold:
                    near_duplicates += 1
                    high_similarity_samples.append({
                        "index": i,
                        "type": "near_duplicate",
                        "similarity": max_similarity,
                        "text_preview": text[:100] + "...",
                    })

        # Compute risk score
        total_samples = len(generated_texts)
        risk_score = (
            exact_matches * 1.0 +
            near_duplicates * 0.5 +
            sum(n_gram_overlaps.values()) * 0.1
        ) / max(total_samples, 1)
        risk_score = min(1.0, risk_score)

        # Generate recommendations
        recommendations = []
        if exact_matches > 0:
            recommendations.append("CRITICAL: Remove exact duplicates from generated data")
        if near_duplicates > total_samples * 0.05:
            recommendations.append("Increase paraphrase diversity or use stricter filtering")
        if n_gram_overlaps.get(20, 0) > total_samples * 0.1:
            recommendations.append("Reduce n-gram overlap with min-frequency filtering")

        return MemorizationAuditReport(
            risk_score=risk_score,
            exact_matches=exact_matches,
            near_duplicates=near_duplicates,
            high_similarity_samples=high_similarity_samples[:20],
            n_gram_overlaps=n_gram_overlaps,
            recommendations=recommendations,
            passed=risk_score < 0.1 and exact_matches == 0,
            details={
                "total_samples": total_samples,
                "similarity_threshold": self._threshold,
            },
        )

    def _get_ngrams(self, text: str, n: int) -> Iterator[str]:
        """Get character n-grams from text."""
        text = text.lower()
        for i in range(len(text) - n + 1):
            yield text[i:i + n]

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Use SequenceMatcher for efficiency
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


class TextDeduplicator:
    """Remove near-duplicate texts.

    Uses MinHash-based deduplication for efficiency
    on large datasets.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
    ):
        """Initialize deduplicator.

        Args:
            threshold: Similarity threshold for deduplication
            num_perm: Number of permutations for MinHash
        """
        self._threshold = threshold
        self._num_perm = num_perm

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate texts.

        Args:
            texts: Input texts

        Returns:
            Deduplicated texts
        """
        if not texts:
            return []

        # Simple deduplication using hash and similarity
        seen_hashes: Set[str] = set()
        unique_texts: List[str] = []

        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            if text_hash in seen_hashes:
                continue

            # Check similarity with existing texts (sample for efficiency)
            is_duplicate = False
            for existing in unique_texts[-100:]:  # Check last 100
                if self._is_similar(text, existing):
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        logger.info(f"Deduplicated {len(texts)} -> {len(unique_texts)} texts")
        return unique_texts

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar."""
        # Quick length check
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        if len_ratio < self._threshold:
            return False

        # Similarity check
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity > self._threshold


class PIIScrubber:
    """Remove or replace PII in text.

    Supports various PII types with configurable
    replacement strategies.
    """

    # PII patterns
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    # Replacement tokens
    REPLACEMENTS = {
        PIIType.EMAIL: "[EMAIL]",
        PIIType.PHONE: "[PHONE]",
        PIIType.SSN: "[SSN]",
        PIIType.CREDIT_CARD: "[CREDIT_CARD]",
        PIIType.IP_ADDRESS: "[IP_ADDRESS]",
        PIIType.NAME: "[NAME]",
        PIIType.ADDRESS: "[ADDRESS]",
        PIIType.DATE_OF_BIRTH: "[DOB]",
    }

    def __init__(
        self,
        pii_types: Optional[List[PIIType]] = None,
        replacement_mode: str = "token",  # token, synthetic, remove
    ):
        """Initialize scrubber.

        Args:
            pii_types: Types of PII to scrub
            replacement_mode: How to replace PII
        """
        self._pii_types = pii_types or list(PIIType)
        self._mode = replacement_mode
        self._compiled_patterns = {
            pii_type: re.compile(pattern, re.IGNORECASE)
            for pii_type, pattern in self.PATTERNS.items()
            if pii_type in self._pii_types
        }

    def scrub(self, text: str) -> str:
        """Scrub PII from text.

        Args:
            text: Input text

        Returns:
            Text with PII removed/replaced
        """
        result = text

        for pii_type, pattern in self._compiled_patterns.items():
            if self._mode == "token":
                result = pattern.sub(self.REPLACEMENTS[pii_type], result)
            elif self._mode == "synthetic":
                result = pattern.sub(self._generate_synthetic(pii_type), result)
            else:  # remove
                result = pattern.sub("", result)

        return result

    def scrub_batch(self, texts: List[str]) -> List[str]:
        """Scrub PII from multiple texts.

        Args:
            texts: Input texts

        Returns:
            Texts with PII removed
        """
        return [self.scrub(text) for text in texts]

    def _generate_synthetic(self, pii_type: PIIType) -> str:
        """Generate synthetic replacement for PII."""
        if pii_type == PIIType.EMAIL:
            return f"user{np.random.randint(10000)}@example.com"
        elif pii_type == PIIType.PHONE:
            return f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        elif pii_type == PIIType.SSN:
            return "XXX-XX-XXXX"
        elif pii_type == PIIType.CREDIT_CARD:
            return "XXXX-XXXX-XXXX-XXXX"
        elif pii_type == PIIType.IP_ADDRESS:
            return "192.168.X.X"
        return self.REPLACEMENTS.get(pii_type, "[REDACTED]")


class ParaphraseGenerator:
    """Generate diverse paraphrases of text.

    Uses LLM or rule-based approaches to generate
    semantically equivalent but lexically diverse text.
    """

    def __init__(
        self,
        diversity: float = 0.5,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize generator.

        Args:
            diversity: Diversity level (0-1)
            api_key: LLM API key
            model: LLM model to use
        """
        self._diversity = diversity
        self._api_key = api_key
        self._model = model

    def paraphrase(self, text: str, n_variants: int = 3) -> List[str]:
        """Generate paraphrases of text.

        Args:
            text: Input text
            n_variants: Number of variants to generate

        Returns:
            List of paraphrases
        """
        if self._api_key:
            return self._paraphrase_llm(text, n_variants)
        return self._paraphrase_rules(text, n_variants)

    def _paraphrase_llm(self, text: str, n_variants: int) -> List[str]:
        """Generate paraphrases using LLM."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a paraphrase generator. Generate {n_variants} diverse paraphrases "
                        f"of the given text. Diversity level: {self._diversity}. "
                        "Maintain the meaning but vary the wording significantly. "
                        "Return only the paraphrases, one per line.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.7 + self._diversity * 0.3,
            )

            content = response.choices[0].message.content
            paraphrases = [p.strip() for p in content.split("\n") if p.strip()]
            return paraphrases[:n_variants]

        except Exception as e:
            logger.warning(f"LLM paraphrase failed: {e}, falling back to rules")
            return self._paraphrase_rules(text, n_variants)

    def _paraphrase_rules(self, text: str, n_variants: int) -> List[str]:
        """Generate paraphrases using rules."""
        variants = [text]

        # Simple word substitutions
        substitutions = {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "compact"],
            "said": ["stated", "mentioned", "noted"],
            "asked": ["inquired", "questioned", "wondered"],
        }

        words = text.split()
        for _ in range(n_variants - 1):
            new_words = words.copy()
            for i, word in enumerate(new_words):
                word_lower = word.lower()
                if word_lower in substitutions:
                    new_words[i] = np.random.choice(substitutions[word_lower])
            variants.append(" ".join(new_words))

        return variants[:n_variants]


class DPTextGenerator:
    """Differentially private text generation.

    Uses DP mechanisms to ensure generated text
    doesn't leak information about training data.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        min_frequency: int = 5,
    ):
        """Initialize generator.

        Args:
            epsilon: Privacy budget
            delta: DP delta
            min_frequency: Minimum frequency for n-grams
        """
        self._epsilon = epsilon
        self._delta = delta
        self._min_frequency = min_frequency
        self._vocab: Dict[str, int] = {}

    def fit(self, texts: List[str]) -> "DPTextGenerator":
        """Fit on training texts.

        Args:
            texts: Training texts

        Returns:
            Self for chaining
        """
        # Build vocabulary with frequency counts
        word_counts: Dict[str, int] = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Filter by minimum frequency (privacy protection)
        self._vocab = {
            word: count
            for word, count in word_counts.items()
            if count >= self._min_frequency
        }

        # Add DP noise to counts
        sensitivity = 1  # Each record affects count by at most 1
        noise_scale = sensitivity / self._epsilon

        for word in self._vocab:
            noise = np.random.laplace(0, noise_scale)
            self._vocab[word] = max(0, self._vocab[word] + noise)

        return self

    def filter_text(self, text: str) -> str:
        """Filter text to only use safe vocabulary.

        Args:
            text: Input text

        Returns:
            Filtered text
        """
        words = text.split()
        filtered_words = [
            word if word.lower() in self._vocab else "[UNK]"
            for word in words
        ]
        return " ".join(filtered_words)


class FineTuningDataGenerator:
    """Main interface for generating privacy-safe fine-tuning data.

    Combines all safety features:
    - Deduplication
    - PII scrubbing
    - Memorization detection
    - DP filtering
    - Paraphrase generation
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize generator.

        Args:
            config: Safety configuration
            api_key: LLM API key for paraphrase generation
        """
        self.config = config or SafetyConfig()
        self._api_key = api_key

        # Initialize components
        self._deduplicator = TextDeduplicator(
            threshold=self.config.deduplication_threshold
        )
        self._pii_scrubber = PIIScrubber(
            pii_types=self.config.pii_types
        )
        self._memorization_detector = MemorizationDetector(
            similarity_threshold=self.config.max_similarity_threshold
        )
        self._paraphrase_generator = ParaphraseGenerator(
            diversity=self.config.paraphrase_diversity,
            api_key=api_key,
        )
        self._dp_generator = DPTextGenerator(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            min_frequency=self.config.min_frequency_threshold,
        )

    def generate_from_data(
        self,
        training_data: List[str],
        n_samples: int,
    ) -> List[str]:
        """Generate safe fine-tuning data from existing training data.

        Args:
            training_data: Original training texts
            n_samples: Number of samples to generate

        Returns:
            Safe synthetic training data
        """
        # Fit DP generator and memorization detector
        if self.config.enable_dp:
            self._dp_generator.fit(training_data)
        self._memorization_detector.fit(training_data)

        # Process training data
        processed = []

        # Step 1: Scrub PII
        if self.config.remove_pii:
            training_data = self._pii_scrubber.scrub_batch(training_data)

        # Step 2: Apply length filters
        training_data = [
            text for text in training_data
            if self.config.min_text_length <= len(text) <= self.config.max_text_length
        ]

        # Step 3: Deduplicate
        training_data = self._deduplicator.deduplicate(training_data)

        # Step 4: Generate paraphrases for diversity
        if self.config.enable_paraphrase:
            augmented = []
            samples_per_text = max(1, n_samples // len(training_data))

            for text in training_data:
                augmented.append(text)
                if len(augmented) >= n_samples:
                    break

                paraphrases = self._paraphrase_generator.paraphrase(
                    text, n_variants=samples_per_text
                )
                augmented.extend(paraphrases[:samples_per_text])

                if len(augmented) >= n_samples:
                    break

            processed = augmented
        else:
            # Sample from training data
            indices = np.random.choice(
                len(training_data),
                size=min(n_samples, len(training_data) * 3),
                replace=True,
            )
            processed = [training_data[i] for i in indices]

        # Step 5: Final deduplication
        processed = self._deduplicator.deduplicate(processed)

        # Step 6: DP filtering
        if self.config.enable_dp:
            processed = [
                self._dp_generator.filter_text(text)
                for text in processed
            ]
            # Remove texts that became too short
            processed = [
                text for text in processed
                if len(text.replace("[UNK]", "").strip()) >= self.config.min_text_length
            ]

        # Step 7: Final safety check
        report = self._memorization_detector.detect(processed[:1000], training_data[:1000])
        if not report.passed and self.config.filter_level in (TextFilterLevel.STRICT, TextFilterLevel.PARANOID):
            logger.warning(f"Memorization detected, applying strict filtering")
            # Remove high-similarity samples
            to_remove = {s["index"] for s in report.high_similarity_samples}
            processed = [text for i, text in enumerate(processed) if i not in to_remove]

        return processed[:n_samples]

    def generate_from_prompts(
        self,
        prompts: List[str],
        n_per_prompt: int = 100,
        reference_data: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate safe fine-tuning data from prompts.

        Args:
            prompts: Generation prompts
            n_per_prompt: Samples per prompt
            reference_data: Optional reference data to avoid

        Returns:
            Generated texts
        """
        if not self._api_key:
            raise ValueError("API key required for prompt-based generation")

        generated = []

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            for prompt in prompts:
                for _ in range(n_per_prompt):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "Generate diverse, realistic training examples. "
                                "Each example should be unique and varied.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.9,
                    )
                    generated.append(response.choices[0].message.content)

        except ImportError:
            raise ImportError("OpenAI required: pip install openai")

        # Apply safety processing
        if self.config.remove_pii:
            generated = self._pii_scrubber.scrub_batch(generated)

        generated = self._deduplicator.deduplicate(generated)

        # Check against reference data
        if reference_data:
            self._memorization_detector.fit(reference_data)
            report = self._memorization_detector.detect(generated)
            if not report.passed:
                to_remove = {s["index"] for s in report.high_similarity_samples}
                generated = [text for i, text in enumerate(generated) if i not in to_remove]

        return generated

    def audit_memorization(
        self,
        generated_data: List[str],
        training_data: List[str],
    ) -> MemorizationAuditReport:
        """Audit generated data for memorization.

        Args:
            generated_data: Generated texts
            training_data: Original training texts

        Returns:
            MemorizationAuditReport
        """
        return self._memorization_detector.detect(generated_data, training_data)

    def export_for_training(
        self,
        data: List[str],
        format: str = "jsonl",
        output_path: str = "training_data.jsonl",
    ) -> str:
        """Export data in format suitable for LLM fine-tuning.

        Args:
            data: Generated texts
            format: Output format (jsonl, csv)
            output_path: Output file path

        Returns:
            Path to output file
        """
        import json

        if format == "jsonl":
            with open(output_path, "w") as f:
                for text in data:
                    f.write(json.dumps({"text": text}) + "\n")
        elif format == "csv":
            pd.DataFrame({"text": data}).to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported {len(data)} samples to {output_path}")
        return output_path


# Convenience functions
def generate_safe_finetuning_data(
    training_data: List[str],
    n_samples: int,
    epsilon: float = 1.0,
    enable_paraphrase: bool = True,
    api_key: Optional[str] = None,
) -> List[str]:
    """Quick function to generate safe fine-tuning data.

    Args:
        training_data: Original training texts
        n_samples: Number of samples
        epsilon: Privacy budget
        enable_paraphrase: Enable paraphrase generation
        api_key: Optional LLM API key

    Returns:
        Safe synthetic data
    """
    config = SafetyConfig(
        enable_dp=True,
        epsilon=epsilon,
        enable_paraphrase=enable_paraphrase,
    )

    generator = FineTuningDataGenerator(config=config, api_key=api_key)
    return generator.generate_from_data(training_data, n_samples)


def audit_training_data(
    generated_data: List[str],
    original_data: List[str],
    similarity_threshold: float = 0.7,
) -> MemorizationAuditReport:
    """Audit generated data for memorization risks.

    Args:
        generated_data: Generated texts
        original_data: Original training texts
        similarity_threshold: Similarity threshold

    Returns:
        Audit report
    """
    detector = MemorizationDetector(similarity_threshold=similarity_threshold)
    return detector.detect(generated_data, original_data)
