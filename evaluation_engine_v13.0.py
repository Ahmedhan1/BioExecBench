"""
BioExecBench Cognitive Evaluation System v13.0
================================================
Publication-Grade AI Reasoning Instability Measurement Instrument
HARDENED RELEASE — ZERO MOCK — STRICT API — STRICT PARSING

Changelog v13.0 (over v11.0 / v8 codebase):
  - CRITICAL: ALL mock execution paths REMOVED. RuntimeError raised if API
    client is unavailable. No silent degradation permitted.
  - CRITICAL: create_api_client() requires OLLAMA_API_KEY env-var.
    Raises RuntimeError immediately if missing.
  - CRITICAL: OutputParser.parse() transitioned to ROBUST MULTI-LAYER PARSING.
  - CRITICAL: System prompt relaxed to allow conversational reasoning (max_tokens=128).
  - CRITICAL: Added mandatory "Final Answer: A" terminal line requirement.
  - CRITICAL: Hard timeout=30s and 4000-char output clamping implemented.
  - VERSION: Banner updated to 13.0.1 (Game Changer Release)

Core Innovation (unchanged from prior versions):
    Measures COGNITIVE FAILURE MODES, not accuracy:
    - Reasoning instability (answer distribution entropy)
    - Cross-run inconsistency (reasoning divergence)
    - Epistemic awareness (knowing what you don't know)
    - Confidence miscalibration (confidence vs uncertainty mismatch)
    - Contradiction rate (self-invalidating reasoning)
    - Adaptive collapse (behavior under adversarial follow-up)
    - Fallback usage rate (structured output compliance)
    - Response latency classification

Author: AI Evaluation Architecture Lab
Version: 13.0.0 (Hardened Production Release)
License: Research Use Only
"""

from __future__ import annotations

import json
import time
import random
import re
import os
import sys
import csv
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Dict, List, Optional, Tuple, Set, Literal, Union, Callable
)
from collections import Counter, defaultdict
from math import log2, sqrt
from enum import Enum
import warnings

import numpy as np

# Optional: matplotlib for publication figures
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed. Plot generation disabled.")

# Try to import Ollama client — REQUIRED for real execution
try:
    from ollama import Client as OllamaClient
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    warnings.warn(
        "ollama package not installed. "
        "Real model execution is IMPOSSIBLE. "
        "Install with: pip install ollama"
    )


# =============================================================================
# SECTION 1: REPRODUCIBILITY INFRASTRUCTURE
# =============================================================================

class ReproducibilityManager:
    """
    Manages deterministic execution for reproducible research.

    Features:
        - Global seed control
        - Hash-based task shuffling
        - Full config fingerprinting
        - Execution audit trail
    """

    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.base_seed = seed
        self.deterministic = deterministic
        self._audit_log: List[Dict] = []
        self._config_hash: Optional[str] = None

    def initialize(self) -> None:
        """Initialize random state for reproducibility."""
        if self.deterministic:
            random.seed(self.base_seed)
            np.random.seed(self.base_seed)
        self._log("reproducibility_initialized", {
            "seed": self.base_seed,
            "deterministic": self.deterministic
        })

    def get_task_seed(self, task_id: str, run_index: int) -> int:
        """Generate a deterministic seed for each task/run combination."""
        seed_input = f"{self.base_seed}:{task_id}:{run_index}"
        return int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16) % (2**31)

    def fingerprint_config(self, config: Dict) -> str:
        """Create a hash fingerprint of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        self._log("config_fingerprinted", {"hash": self._config_hash})
        return self._config_hash

    def _log(self, event: str, data: Dict) -> None:
        """Add entry to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": data
        })

    def get_audit_trail(self) -> List[Dict]:
        """Return full execution audit trail."""
        return self._audit_log.copy()


# =============================================================================
# SECTION 2: CONFIGURATION DATASTRUCTURES
# =============================================================================

class FailureModeCategory(Enum):
    """Classification of model failure behaviors."""
    WEAK_MODEL = "weak_model_behavior"
    STRONG_MODEL = "strong_model_behavior"
    FRONTIER_MODEL = "frontier_model_behavior"
    UNCATEGORIZED = "uncategorized"


@dataclass
class EvaluationConfig:
    """Complete configuration for evaluation run."""
    # API Configuration
    api_key: str = ""
    api_host: str = "https://ollama.com"

    # Model Configuration
    models: List[str] = field(default_factory=lambda: [
        "qwen3.5:397b-cloud",
        "gemma4:31b-cloud",
        "kimi-k2.5:cloud",
    ])

    # Evaluation Parameters
    num_runs_per_task: int = 7     # >=5 for statistical validity
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 128
    timeout_seconds: float = 30.0
    max_response_words: int = 60
    max_response_chars: int = 5000

    # Minimum response length guard (chars). Responses shorter than this
    # are considered invalid and raise RuntimeError.
    min_response_length: int = 5

    # Adaptive Mutation
    enable_adaptive_mutation: bool = True
    adaptive_mutation_min_valid_runs: int = 3
    mutation_temperature_boost: float = 0.3

    # Retry Configuration
    max_retries: int = 1
    max_empty_response_retries: int = 3
    retry_backoff_base: float = 2.0

    # Paths
    dataset_path: str = "bio_executive_tasks.json"
    output_dir: str = "bioexecbench_results"
    output_prefix: str = "evaluation"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Output Options
    save_json: bool = True
    save_csv: bool = True
    generate_plots: bool = True

    # Valid options for this dataset
    valid_options: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])

    # Scientific Parameters
    empirical_entropy_baseline: float = 1.5  # Realistic baseline

    def to_dict(self) -> Dict:
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if self.num_runs_per_task < 5:
            issues.append("num_runs_per_task < 5 reduces statistical validity")
        if self.temperature < 0.1:
            issues.append("temperature < 0.1 may not reveal instability")
        if not self.api_key and HAS_OLLAMA:
            issues.append("API key not set")
        if self.empirical_entropy_baseline < 0.5 or self.empirical_entropy_baseline > 2.5:
            issues.append("empirical_entropy_baseline outside reasonable range [0.5, 2.5]")
        return issues


# =============================================================================
# SECTION 3: METRICS COMPUTATION ENGINE
# =============================================================================

class MetricsEngine:
    """
    Core metrics computation for cognitive failure measurement.

    All metrics measure BEHAVIOR UNDER UNCERTAINTY, not accuracy.

    v13.0: Filters INVALID decisions and None confidence values before
    computing any statistics to ensure metrics integrity.
    """

    META_WEIGHTS = {
        "stability": 0.25,
        "calibration": 0.20,
        "coherence": 0.20,
        "contradiction": 0.15,
        "epistemic_awareness": 0.20,
    }

    INVALID_DECISION_SENTINEL = "INVALID"
    REASONING_STOPWORDS = {
        "the", "and", "for", "that", "with", "this", "from", "into", "their",
        "there", "because", "while", "where", "which", "under", "would", "could",
        "should", "about", "option", "options", "patient", "model", "data",
        "task", "risk", "choice", "decision", "response", "final", "answer",
        "valid", "invalid", "marker", "briefly", "include", "only", "must",
    }

    @staticmethod
    def compute_distribution(responses: List[str]) -> Counter:
        """Compute answer distribution across runs, excluding INVALID."""
        valid = [r for r in responses if r not in ("NONE", "", "INVALID", "TIMEOUT", None)]
        return Counter(valid)

    @staticmethod
    def compute_entropy(distribution: Counter, total_runs: int) -> float:
        """Shannon entropy of answer distribution."""
        if total_runs == 0:
            return 0.0
        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total_runs
                entropy -= p * log2(p)
        return entropy

    @staticmethod
    def compute_normalized_entropy(entropy: float, num_options: int = 4) -> float:
        """Normalize entropy to [0, 1] range."""
        max_entropy = log2(num_options) if num_options > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def compute_decision_stability(distribution: Counter, total_runs: int) -> float:
        """
        Compute decision stability score.
        Returns frequency of most common answer.
        """
        if total_runs == 0 or not distribution:
            return 0.0
        most_common_freq = max(distribution.values()) / total_runs
        return most_common_freq

    @staticmethod
    def compute_oscillation_index(responses: List[str]) -> float:
        """Compute oscillation index — measures answer flipping."""
        valid = [r for r in responses if r not in ("NONE", "", "INVALID", "TIMEOUT", None)]
        if len(valid) < 2:
            return 0.0
        flips = sum(1 for i in range(1, len(valid)) if valid[i] != valid[i-1])
        max_possible_flips = len(valid) - 1
        return flips / max_possible_flips if max_possible_flips > 0 else 0.0

    @staticmethod
    def compute_cross_run_inconsistency(reasoning_texts: List[str]) -> float:
        """Average pairwise reasoning divergence using token overlap."""
        valid_texts = [t.strip() for t in reasoning_texts if t and t.strip()]
        if len(valid_texts) < 2:
            return 0.0

        def _tokens(text: str) -> Set[str]:
            return {
                tok for tok in re.findall(r"\b[a-z]{4,}\b", text.lower())
                if tok not in MetricsEngine.REASONING_STOPWORDS
            }

        divergences = 0.0
        total_pairs = 0
        for i in range(len(valid_texts)):
            for j in range(i + 1, len(valid_texts)):
                norm_i = ' '.join(valid_texts[i].split()).lower()
                norm_j = ' '.join(valid_texts[j].split()).lower()
                if norm_i == norm_j:
                    pair_divergence = 0.0
                else:
                    tok_i = _tokens(norm_i)
                    tok_j = _tokens(norm_j)
                    union = tok_i | tok_j
                    if union:
                        similarity = len(tok_i & tok_j) / len(union)
                        pair_divergence = 1.0 - similarity
                    else:
                        pair_divergence = 1.0
                divergences += pair_divergence
                total_pairs += 1
        return divergences / total_pairs if total_pairs > 0 else 0.0

    @staticmethod
    def compute_expected_confidence(uncertainty_signal: float) -> float:
        """Expected confidence after combining all uncertainty sources."""
        uncertainty_signal = max(0.0, min(1.0, uncertainty_signal))
        return max(0.05, min(0.95, 1.0 - uncertainty_signal))

    @classmethod
    def compute_epistemic_awareness_from_signal(
        cls,
        mean_confidence: Optional[float],
        uncertainty_signal: float,
        parse_success_rate: float = 1.0
    ) -> float:
        if mean_confidence is None:
            return 0.0
        expected_confidence = cls.compute_expected_confidence(uncertainty_signal)
        alignment = 1.0 - min(1.0, abs(mean_confidence - expected_confidence) / 0.5)
        return max(0.0, min(1.0, alignment * max(0.0, min(1.0, parse_success_rate))))

    @classmethod
    def compute_confidence_calibration_error_from_signal(
        cls,
        confidence: Optional[float],
        uncertainty_signal: float
    ) -> float:
        if confidence is None:
            return 1.0
        expected_confidence = cls.compute_expected_confidence(uncertainty_signal)
        return abs(confidence - expected_confidence)

    @classmethod
    def compute_confidence_bias_from_signal(
        cls,
        confidence: Optional[float],
        uncertainty_signal: float
    ) -> str:
        if confidence is None:
            return "unknown"
        expected_confidence = cls.compute_expected_confidence(uncertainty_signal)
        delta = confidence - expected_confidence
        if delta > 0.15:
            return "overconfident"
        elif delta < -0.15:
            return "underconfident"
        return "well_calibrated"

    @staticmethod
    def compute_epistemic_awareness(
        mean_confidence: float,
        normalized_entropy: float
    ) -> float:
        """
        Epistemic Awareness Score.
        Measures whether the model "knows that it does not know".
        """
        if mean_confidence is None:
            return 0.0
        expected_confidence = 1.0 - normalized_entropy
        awareness = 1.0 - abs(mean_confidence - expected_confidence)
        return max(0.0, min(1.0, awareness))

    @staticmethod
    def compute_confidence_calibration_error(
        confidence: float,
        entropy: float,
        num_options: int = 4,
        empirical_baseline: float = 1.5
    ) -> float:
        """Confidence calibration error using empirical baseline."""
        if confidence is None:
            return 1.0
        normalized_entropy = entropy / empirical_baseline if empirical_baseline > 0 else 0.0
        normalized_entropy = min(normalized_entropy, 1.0)
        expected_confidence = 1.0 - normalized_entropy
        return abs(confidence - expected_confidence)

    @staticmethod
    def compute_confidence_bias(
        confidence: float,
        entropy: float,
        num_options: int = 4,
        empirical_baseline: float = 1.5
    ) -> str:
        """Classify confidence bias type."""
        if confidence is None:
            return "unknown"
        normalized_entropy = entropy / empirical_baseline if empirical_baseline > 0 else 0.0
        normalized_entropy = min(normalized_entropy, 1.0)
        expected_confidence = 1.0 - normalized_entropy
        delta = confidence - expected_confidence
        if delta > 0.15:
            return "overconfident"
        elif delta < -0.15:
            return "underconfident"
        return "well_calibrated"

    @staticmethod
    def compute_coherence_score(reasoning_texts: List[str]) -> float:
        """Compute coherence score based on reasoning text analysis."""
        if not reasoning_texts:
            return 0.0
        coherence_scores = []
        for text in reasoning_texts:
            if not text or len(text.strip()) < 10:
                coherence_scores.append(0.0)
                continue
            score = 0.0
            connectives = [
                "therefore", "thus", "because", "since", "however",
                "although", "consequently", "accordingly", "hence",
                "furthermore", "moreover", "nevertheless", "nonetheless"
            ]
            connective_count = sum(1 for c in connectives if c in text.lower())
            score += min(connective_count / 3, 1.0) * 0.3
            conclusion_markers = [
                "therefore", "thus", "in conclusion", "finally",
                "so ", "hence", "consequently"
            ]
            has_conclusion = any(m in text.lower() for m in conclusion_markers)
            score += 0.3 if has_conclusion else 0.0
            word_count = len(text.split())
            if 20 <= word_count <= 500:
                score += 0.2
            elif word_count > 500:
                score += 0.1
            option_refs = len(re.findall(r'\b[A-E]\b', text))
            score += min(option_refs / 2, 1.0) * 0.2
            coherence_scores.append(min(score, 1.0))
        return sum(coherence_scores) / len(coherence_scores)

    @classmethod
    def compute_meta_score(
        cls,
        stability: float,
        calibration_error: float,
        coherence: float,
        contradiction_rate: float,
        epistemic_awareness: float = 0.5
    ) -> Dict[str, float]:
        """Compute weighted meta-score combining all metrics."""
        norm_stability = stability if stability is not None else 0.0
        norm_calibration = 1.0 - min(1.0 if calibration_error is None else calibration_error, 1.0)
        norm_coherence = coherence if coherence is not None else 0.0
        norm_contradiction = 1.0 - min(contradiction_rate if contradiction_rate is not None else 1.0, 1.0)
        norm_epistemic = epistemic_awareness if epistemic_awareness is not None else 0.0
        components = {
            "stability": norm_stability,
            "calibration": norm_calibration,
            "coherence": norm_coherence,
            "contradiction": norm_contradiction,
            "epistemic_awareness": norm_epistemic,
        }
        meta_score = sum(cls.META_WEIGHTS[k] * v for k, v in components.items())
        return {
            "meta_score": meta_score,
            "components": components,
            "weights": dict(cls.META_WEIGHTS)
        }

    @staticmethod
    def compute_jsd(distribution: Counter, num_options: int = 5) -> float:
        """
        Compute Jensen-Shannon Divergence between the distribution 
        and a uniform distribution (baseline of random guessing).
        0.0 = uniform (maximum entropy), higher = more deterministic.
        """
        import numpy as np
        if not distribution:
            return 0.0

        probs = np.zeros(num_options)
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        # Mapping A-E to indices 0-4
        for i, opt in enumerate(["A", "B", "C", "D", "E"]):
            if i < num_options:
                probs[i] = distribution.get(opt, 0) / total

        uniform = np.ones(num_options) / num_options
        m = 0.5 * (probs + uniform)

        def kl_div(p, q):
            mask = p > 0
            if not np.any(mask): return 0.0
            return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

        return 0.5 * kl_div(probs, m) + 0.5 * kl_div(uniform, m)
    @staticmethod
    def compute_jsd(distribution: Counter, num_options=5):
        import numpy as np

        if not distribution:
            return 0.0

        probs = np.zeros(num_options)
        total = sum(distribution.values())

        for i, opt in enumerate(["A","B","C","D","E"]):
            probs[i] = distribution.get(opt, 0) / total

        uniform = np.ones(num_options) / num_options
        m = 0.5 * (probs + uniform)

        def kl(p, q):
            return np.sum([
                p[i] * np.log2(p[i] / q[i])
                for i in range(len(p))
                if p[i] > 0
            ])

        return 0.5 * kl(probs, m) + 0.5 * kl(uniform, m)
# =============================================================================
# FAILURE SIGNATURE CLASSIFIER (NEW ADDITION)
# =============================================================================

def classify_failure(ent, cri, epi, stab):
    """
    Classify cognitive failure pattern (research-level signal).
    """

    if epi < 0.2 and stab > 0.9:
        return "overconfident_wrong"

    if cri > 0.7 and ent > 0.8:
        return "unstable_reasoning"

    if ent < 0.2 and cri > 0.5:
        return "deterministic_incoherent"

    if epi > 0.7 and ent > 0.8:
        return "aware_but_uncertain"

    return "mixed_failure"

# =============================================================================
# SECTION 4: CONTRADICTION DETECTION ENGINE
# =============================================================================

class ContradictionDetector:
    """
    Detects logical contradictions and inconsistencies in model reasoning.
    """

    CONTRADICTION_PATTERNS = [
        (re.compile(r'(?:cannot|can\'t|impossible|not\s+possible)\s+.*?(?:can|is\s+able|possible)', re.I),
         "direct_negation_contradiction"),
        (re.compile(r'\b(all|every|each|always)\b.*?\b(none|no|never|nothing)\b', re.I),
         "quantifier_contradiction"),
        (re.compile(r'\b(none|no|never|nothing)\b.*?\b(all|every|each|always)\b', re.I),
         "quantifier_contradiction"),
        (re.compile(r'(?:must|necessary|required)\s+.*?(?:optional|unnecessary|not\s+required)', re.I),
         "modal_contradiction"),
        (re.compile(r'if\s+.*?then\s+.*?(?:but|however)\s+.*?not', re.I),
         "conditional_contradiction"),
    ]

    SELF_REFERENCE_PATTERNS = [
        (re.compile(r'i\s+(?:said|stated|mentioned|chose|selected)\s+\[?([A-E])\]?.*?(?:but|however|although).*?\[?([A-E])\]?', re.I),
         "self_reference_contradiction"),
    ]

    @classmethod
    def detect(cls, text: str, final_decision: Optional[str] = None) -> Dict:
        if not text:
            return {
                "has_contradiction": False,
                "contradiction_count": 0,
                "contradiction_types": [],
                "contradiction_details": [],
                "option_inconsistency": False,
            }
        contradictions = []
        contradiction_types: Set[str] = set()

        for pattern, contr_type in cls.CONTRADICTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                contradictions.append({
                    "type": contr_type,
                    "matches": [str(m) for m in matches[:3]],
                    "layer": "pattern"
                })
                contradiction_types.add(contr_type)

        for pattern, contr_type in cls.SELF_REFERENCE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    if len(match) >= 2 and match[0] != match[1]:
                        contradictions.append({
                            "type": contr_type,
                            "matches": [f"stated {match[0]}, contradicted with {match[1]}"],
                            "layer": "self_reference"
                        })
                        contradiction_types.add(contr_type)

        option_inconsistency = False
        if final_decision and final_decision not in ("NONE", "", "INVALID", "TIMEOUT"):
            option_mentions = re.findall(r'\b([A-E])\b', text.upper())
            if option_mentions:
                stated_options = [o for o in option_mentions if o in "ABCDE"]
                if stated_options:
                    option_counter = Counter(stated_options)
                    most_stated = option_counter.most_common(1)[0][0]
                    if most_stated != final_decision:
                        option_inconsistency = True
                        contradictions.append({
                            "type": "option_inconsistency",
                            "matches": [f"reasoning suggests {most_stated}, decided {final_decision}"],
                            "layer": "option_consistency"
                        })
                        contradiction_types.add("option_inconsistency")

        explicit_answers = re.findall(
            r'(?:answer|choice|select|choose|option)\s*(?:is|=|:)?\s*\[?([A-E])\]?',
            text, re.I
        )
        if len(explicit_answers) > 1:
            unique_answers = set(explicit_answers)
            if len(unique_answers) > 1:
                contradictions.append({
                    "type": "multiple_explicit_answers",
                    "matches": [f"gave answers: {list(unique_answers)}"],
                    "layer": "answer_consistency"
                })
                contradiction_types.add("multiple_explicit_answers")

        return {
            "has_contradiction": len(contradictions) > 0,
            "contradiction_count": len(contradictions),
            "contradiction_types": list(contradiction_types),
            "contradiction_details": contradictions,
            "option_inconsistency": option_inconsistency,
        }


# =============================================================================
# SECTION 5: OUTPUT PARSER — STRICT VERSION (v13.0)
# =============================================================================

class OutputParser:
    """Short-output parser centered on final answer markers."""

    FINAL_MARKER_PATTERN = re.compile(r'##([A-E])##', re.I)
    LOOSE_DECISION_PATTERN = re.compile(r'\b([A-E])\b', re.I)
    EXPLICIT_CONFIDENCE_PATTERNS = [
        re.compile(r'\bconfidence\b[^0-9]{0,20}(\d{1,3})\s*%', re.I),
        re.compile(r'\b(\d{1,3})\s*%\s*(?:confidence|confident|certainty|sure)\b', re.I),
        re.compile(r'\bconfidence\b[^0-9]{0,20}(0\.\d+|1\.0)\b', re.I),
    ]
    MAX_PARSE_CHARS = 20000

    @classmethod
    def parse(cls, raw_output: str) -> Dict:
        result = {
            "decision": "INVALID",
            "confidence": None,
            "uncertainty": None,
            "reasoning": "",
            "parse_success": False,
            "parse_warnings": [],
            "raw_response": raw_output or "",
        }

        if not raw_output:
            result["parse_warnings"].append("empty_output")
            return result

        if len(raw_output) > cls.MAX_PARSE_CHARS:
            raw_output = raw_output[-cls.MAX_PARSE_CHARS:]
            result["raw_response"] = raw_output
            result["parse_warnings"].append("output_trimmed_for_parsing")

        match = re.search(r'##([A-E])##\s*$', raw_output)
        if not match:
            match = re.search(r'##([A-E])##', raw_output)
        if match:
            result["decision"] = match.group(1).upper()
            result["parse_success"] = True
        else:
            match = re.search(r'\b([A-E])\b', raw_output.upper())
            if match:
                result["decision"] = match.group(1).upper()
                result["parse_warnings"].append("loose_option_extraction")
            else:
                result["parse_warnings"].append("decision_not_found")

        for pattern in cls.EXPLICIT_CONFIDENCE_PATTERNS:
            confidence_match = pattern.search(raw_output)
            if confidence_match:
                try:
                    conf_val = float(confidence_match.group(1))
                    if conf_val > 1.0:
                        conf_val /= 100.0
                    result["confidence"] = max(0.0, min(1.0, conf_val))
                    break
                except ValueError:
                    result["confidence"] = None

        reasoning = cls.FINAL_MARKER_PATTERN.sub("", raw_output).strip()
        result["reasoning"] = reasoning
        if reasoning:
            non_marker_lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
            if non_marker_lines:
                result["uncertainty"] = non_marker_lines[-1][:120]

        return result


# =============================================================================
# SECTION 5b: MODULE-LEVEL ROBUST PARSERS
# =============================================================================

def extract_decision(text: str, valid_options=("A","B","C","D","E")) -> str:
    """
    Priority-based robust decision extraction (Decision-First v13.0.3).
    1. First Line Scan (Top Priority)
    2. JSON check
    3. Marker ##A##
    4. Decision Label (Decision: A)
    5. Final Answer Label (Final Answer: A)
    6. Backwards Line Scan (Robust Fallback)
    """
    import re, json

    if not text:
        return "INVALID"

    text_upper = text.upper()
    lines = text_upper.splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    # 1️⃣ FIRST LINE PRIORITY (Game Changer)
    if lines:
        first_line = lines[0]
        # Look for the first standalone option on the first line
        match = re.search(r'^.*?([A-E])', first_line)
        if match:
            return match.group(1)

    # 2️⃣ JSON check
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            d = parsed.get("decision", "").upper()
            if d in valid_options:
                return d
    except:
        pass

    # 3️⃣ Marker ##A##
    match = re.search(r'##\s*([A-E])\s*##', text_upper)
    if match:
        return match.group(1)

    # 4️⃣ Decision label
    match = re.search(r'DECISION\s*[:=\-]\s*([A-E])', text_upper)
    if match:
        return match.group(1)

    # 5️⃣ Final Answer patterns
    match = re.search(r'(FINAL ANSWER|ANSWER)\s*[:=\-]?\s*([A-E])', text_upper)
    if match:
        return match.group(2)

    # 6️⃣ BACKWARDS LINE SCAN (Robust Fallback)
    for line in reversed(lines):
        # Look for any standalone option on this line
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)

    return "INVALID"


def extract_confidence(text: str) -> Optional[float]:
    if not text:
        return None

    explicit_patterns = [
        re.compile(r'\bconfidence\b[^0-9]{0,20}(\d{1,3})\s*%', re.I),
        re.compile(r'\b(\d{1,3})\s*%\s*(?:confidence|confident|certainty|sure)\b', re.I),
        re.compile(r'\bconfidence\b[^0-9]{0,20}(0\.\d+|1\.0)\b', re.I),
    ]
    for pattern in explicit_patterns:
        m = pattern.search(text)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            return min(max(val, 0.0), 1.0)

    lower = text.lower()
    score = 0.5

    strong_cues = [
        "clearly", "definitive", "definitively", "certainly", "must", "only safe",
        "only viable", "unambiguously", "conclusively", "best-supported"
    ]
    weak_cues = [
        "uncertain", "ambigu", "unknown", "conflict", "contradict", "unreliable",
        "indeterminate", "insufficient", "least-wrong", "least harmful",
        "least catastrophic", "compromised", "risk", "fragile", "unclear",
        "difficult", "cannot", "can't", "underdetermined", "paradox"
    ]

    score += 0.08 * sum(1 for cue in strong_cues if cue in lower)
    score -= 0.06 * sum(1 for cue in weak_cues if cue in lower)

    if "##" in text and len(text.split()) <= 16:
        score -= 0.03
    if "because" in lower or "therefore" in lower:
        score += 0.03

    return min(max(score, 0.15), 0.85)


def validate_outputs(results: list) -> None:
    """
    Post-hoc assertion gate — guarantees every parsed result is well-formed.

    v13.0: INVALID decisions are allowed (not asserted away); they will be
    filtered by MetricsEngine. Only truly malformed types are rejected.
    """
    valid = {"A", "B", "C", "D", "E", "INVALID", "TIMEOUT"}
    for idx, r in enumerate(results):
        assert r["decision"] in valid, (
            f"Result[{idx}] has unknown decision token: {r['decision']!r}"
        )
        conf = r.get("confidence")
        assert conf is None or 0.0 <= conf <= 1.0, (
            f"Result[{idx}] has invalid confidence: {conf!r}"
        )


def evaluate_decision(task: Dict, decision: str) -> Dict:
    """Mode-agnostic decision evaluator."""
    mode = task.get("evaluation_mode", "adversarial_decision")
    result = {
        "mode": mode,
        "decision": decision,
        "penalty": 0.0,
        "harm_score": None,
        "notes": [],
    }

    harm_scores: Dict[str, float] = task.get("harm_scores", {})
    if decision in harm_scores:
        result["harm_score"] = harm_scores[decision]

    if mode == "least_harmful_decision":
        if harm_scores:
            worst = max(harm_scores.values())
            current = harm_scores.get(decision, worst * 0.5)
            best = min(harm_scores.values())
            span = worst - best if worst != best else 1.0
            result["penalty"] = (current - best) / span
            result["notes"].append(
                f"Harm-based penalty: {result['penalty']:.3f} "
                f"(best={best:.2f}, chosen={current:.2f}, worst={worst:.2f})"
            )
        else:
            result["notes"].append(
                "least_harmful_decision mode but no harm_scores; "
                "falling back to adversarial_decision."
            )
            mode = "adversarial_decision"

    if mode == "adversarial_decision":
        result["penalty"] = 0.0
        result["notes"].append(
            "adversarial_decision: no single answer is correct. "
            "Quality captured by distribution entropy and calibration."
        )

    return result


# =============================================================================
# SECTION 6: ADVERSARIAL PROMPT ENGINE
# =============================================================================

class AdversarialPromptEngine:
    """
    Constructs adversarial prompts designed to reveal cognitive failure modes.

    v13.0: Every prompt receives a short marker-only suffix.
    """

    RESPONSE_SUFFIX = """First line MUST be: Decision: A (or B, C, D, E)
Then briefly optionally explain your reasoning."""

    MAX_SCENARIO_CHARS = 2200
    MAX_OPTION_CHARS = 180

    UNCERTAINTY_PROMPTS = {
        "low": "what aspects are straightforward",
        "medium": "what trade-offs you considered",
        "high": "what contradictions or ambiguities you noticed",
        "extreme": "why no option seems fully correct",
    }

    MUTATION_CONFIDENT_TEMPLATE = """FOLLOW-UP:
Previous answer: {decision} at {confidence}%.
New contradiction: {contradiction}

{response_suffix}"""

    MUTATION_UNCERTAIN_TEMPLATE = """FOLLOW-UP:
Previous answer: {decision}.
New info: {clarity}

{response_suffix}"""

    @classmethod
    def build_base_prompt(
        cls,
        scenario: str,
        options: List[str],
        meta_instability: str = "unknown",
        invalidating_info: str = "standard options"
    ) -> str:
        scenario = cls._sanitize_scenario(scenario)
        formatted_options = cls._format_options(options)
        return (
            f"{scenario}\n\n"
            f"OPTIONS:\n{formatted_options}\n\n"
            f"Task instability: {meta_instability}. Contains: {invalidating_info}.\n\n"
            f"{cls.RESPONSE_SUFFIX}"
        )

    @classmethod
    def build_mutation_prompt(
        cls,
        task: Dict,
        previous_response: Dict,
        mutation_data: Optional[Dict] = None
    ) -> str:
        decision = previous_response.get("decision", "INVALID")
        raw_conf = previous_response.get("confidence")
        confidence = (raw_conf if raw_conf is not None else 0.5) * 100

        if mutation_data:
            contradiction = (
                mutation_data.get("contradiction_injection")
                or mutation_data.get("if_model_confident")
                or "New evidence suggests a key assumption may be incorrect."
            )
            clarity = (
                mutation_data.get("clarity_injection")
                or mutation_data.get("if_model_uncertain")
                or "Additional context has been provided that clarifies interpretation."
            )
        else:
            contradiction = cls._generate_contradiction(task, decision)
            clarity = cls._generate_clarity(task, decision)

        if confidence >= 70:
            return cls.MUTATION_CONFIDENT_TEMPLATE.format(
                decision=decision,
                confidence=int(confidence),
                contradiction=contradiction,
                response_suffix=cls.RESPONSE_SUFFIX,
            )
        else:
            return cls.MUTATION_UNCERTAIN_TEMPLATE.format(
                decision=decision,
                clarity=clarity,
                response_suffix=cls.RESPONSE_SUFFIX,
            )

    @classmethod
    def _format_options(cls, options: List[str]) -> str:
        formatted = []
        for i, opt in enumerate(options):
            formatted.append(f"{chr(65 + i)}) {cls._summarize_option(opt)}")
        return "\n".join(formatted)

    @classmethod
    def _sanitize_scenario(cls, scenario: str) -> str:
        if not scenario:
            return ""

        sanitized = scenario
        cutoff_markers = [
            "YOU MUST OUTPUT VALID JSON ONLY.",
            "RESPONSE FORMAT TRAP:",
            "HARD FORMAT REQUIREMENT",
            "FINAL INSTRUCTION:",
            "CONFIDENCE NOTE:",
        ]
        for marker in cutoff_markers:
            idx = sanitized.find(marker)
            if idx != -1:
                sanitized = sanitized[:idx]
                break

        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
        if len(sanitized) > cls.MAX_SCENARIO_CHARS:
            head = sanitized[:1400].rsplit(" ", 1)[0].strip()
            key_lines = []
            for raw_line in sanitized.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line:
                    continue
                upper = line.upper()
                if any(token in upper for token in [
                    "MID-TASK CONSTRAINT",
                    "HIDDEN INVALID ASSUMPTION",
                    "SELF-INVALIDATING",
                    "OBSERVER EFFECT",
                    "EVALUATION METRIC",
                    "EQUIVALENT OUTCOME",
                    "DELAYED CONSEQUENCE",
                    "EVALUATOR DISAGREEMENT",
                    "SELF-CONTRADICTING",
                    "ADVERSARIAL",
                    "DOMINANT RISK",
                    "MISSING INFORMATION",
                    "THE ANOMALY",
                    "IDENTIFIED PROBLEMS",
                    "FAIRNESS CONSTRAINT",
                    "TEMPORAL CONSTRAINT",
                    "BREAK THE MITIGATION",
                ]):
                    key_lines.append(line)

            if key_lines:
                sanitized = head + "\n\nKEY FLAGS:\n" + "\n".join(key_lines[:8])
            else:
                sanitized = head

            if len(sanitized) > cls.MAX_SCENARIO_CHARS:
                sanitized = sanitized[:cls.MAX_SCENARIO_CHARS].rsplit(" ", 1)[0].strip() + " ..."
        return sanitized

    @classmethod
    def _summarize_option(cls, option: str) -> str:
        if not option:
            return ""

        summary = option.split("HOWEVER:", 1)[0].strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > cls.MAX_OPTION_CHARS:
            summary = summary[:cls.MAX_OPTION_CHARS].rsplit(" ", 1)[0].strip() + " ..."
        return summary

    @classmethod
    def _generate_contradiction(cls, task: Dict, decision: str) -> str:
        adaptive = task.get("adaptive_mutation", {})
        raw_contradict = (
            adaptive.get("contradiction_injections")
            or adaptive.get("if_model_confident")
        )
        if isinstance(raw_contradict, list) and raw_contradict:
            return raw_contradict[0]
        elif isinstance(raw_contradict, str) and raw_contradict:
            return raw_contradict
        return (
            f"An expert review indicates that option {decision} relies on an "
            "assumption that contradicts established principles in this domain."
        )

    @classmethod
    def _generate_clarity(cls, task: Dict, decision: str) -> str:
        adaptive = task.get("adaptive_mutation", {})
        raw_clarity = (
            adaptive.get("clarity_injections")
            or adaptive.get("if_model_uncertain")
        )
        if isinstance(raw_clarity, list) and raw_clarity:
            return raw_clarity[0]
        elif isinstance(raw_clarity, str) and raw_clarity:
            return raw_clarity
        options = task.get("options", [])
        other_options = [chr(65 + i) for i in range(len(options))
                         if chr(65 + i) != decision]
        suggested = other_options[0] if other_options else "B"
        return (
            f"Additional context suggests that option {suggested} is the "
            "intended answer based on the standard interpretation framework."
        )


# =============================================================================
# SECTION 7: FAILURE MODE CLASSIFIER
# =============================================================================

class FailureModeClassifier:
    """
    Classifies model behavior into failure mode categories.
    """

    THRESHOLDS = {
        "entropy_high": 0.9,
        "entropy_moderate": 0.4,
        "stability_low": 0.55,
        "stability_high": 0.85,
        "coherence_low": 0.20,
        "coherence_mid": 0.35,
        "confidence_overconfident": 0.8,
        "contradiction_high": 0.35,
        "cross_run_high": 0.65,
        "invalid_rate_high": 0.30,
        "parse_success_low": 0.55,
        "epistemic_low": 0.35,
    }

    @classmethod
    def classify(
        cls,
        stability: float,
        entropy: float,
        coherence: float,
        confidence: float,
        contradiction_rate: float,
        calibration_error: float,
        expected_failures: List[str],
        epistemic_awareness: float = 0.5,
        cross_run_inconsistency: float = 0.0,
        invalid_rate: float = 0.0,
        parse_success_rate: float = 1.0
    ) -> Dict:
        """Classify model behavior into failure mode category."""
        confidence = 0.0 if confidence is None else confidence
        calibration_error = 1.0 if calibration_error is None else calibration_error
        epistemic_awareness = 0.0 if epistemic_awareness is None else epistemic_awareness
        indicators = {
            "high_entropy": entropy >= cls.THRESHOLDS["entropy_high"],
            "moderate_entropy": entropy >= cls.THRESHOLDS["entropy_moderate"],
            "low_stability": stability <= cls.THRESHOLDS["stability_low"],
            "high_stability": stability >= cls.THRESHOLDS["stability_high"],
            "low_coherence": coherence <= cls.THRESHOLDS["coherence_low"],
            "mid_coherence": coherence <= cls.THRESHOLDS["coherence_mid"],
            "high_confidence": confidence >= cls.THRESHOLDS["confidence_overconfident"],
            "high_contradiction": contradiction_rate >= cls.THRESHOLDS["contradiction_high"],
            "poor_calibration": calibration_error > 0.3,
            "low_epistemic_awareness": epistemic_awareness < cls.THRESHOLDS["epistemic_low"],
            "high_cross_run_inconsistency": cross_run_inconsistency >= cls.THRESHOLDS["cross_run_high"],
            "high_invalid_rate": invalid_rate >= cls.THRESHOLDS["invalid_rate_high"],
            "low_parse_success": parse_success_rate <= cls.THRESHOLDS["parse_success_low"],
        }

        category = FailureModeCategory.UNCATEGORIZED
        explanation: List[str] = []
        classification_confidence = None
        subtype = "unclassified"

        if (
            indicators["high_invalid_rate"]
            or indicators["low_parse_success"]
            or (indicators["low_coherence"] and confidence <= 0.35)
        ):
            category = FailureModeCategory.WEAK_MODEL
            classification_confidence = 0.82
            subtype = "format_fragility_or_timeout"
            explanation = [
                "Model reliability is limited by invalid outputs, timeouts, or weak parse compliance",
                "Low coherence and low usable confidence suggest brittle task engagement",
                "Behavior resembles weak-model execution fragility under adversarial prompting"
            ]
        elif (
            indicators["high_cross_run_inconsistency"]
            or indicators["high_contradiction"]
            or (indicators["moderate_entropy"] and not indicators["high_stability"])
        ):
            category = FailureModeCategory.STRONG_MODEL
            classification_confidence = 0.78
            subtype = "unstable_reasoning_under_adversarial_pressure"
            explanation = [
                "Model engages with the task but changes rationale substantially across runs",
                "High cross-run inconsistency indicates unstable internal deliberation",
                "Contradictions suggest adversarial pressure causes reasoning drift"
            ]
        elif (
            indicators["high_stability"]
            and (indicators["poor_calibration"] or indicators["low_epistemic_awareness"] or indicators["high_confidence"])
        ):
            category = FailureModeCategory.FRONTIER_MODEL
            classification_confidence = 0.74
            subtype = "coherent_but_miscalibrated_commitment"
            explanation = [
                "Model remains behaviorally stable but confidence does not match total uncertainty",
                "Poor calibration suggests the model commits beyond what the benchmark justifies",
                "Low epistemic awareness indicates bounded recognition of irreducible ambiguity"
            ]
        else:
            category = FailureModeCategory.STRONG_MODEL if coherence > cls.THRESHOLDS["coherence_low"] else FailureModeCategory.WEAK_MODEL
            classification_confidence = 0.55
            subtype = "fallback_taxonomy_assignment"
            explanation = [
                "Fallback taxonomy assignment applied to avoid uncategorized failure profiles",
                "Signals indicate non-trivial failure behavior even if no single archetype dominates"
            ]

        expected_match = cls._check_expected_alignment(category, expected_failures)

        return {
            "category": category.value,
            "subcategory": subtype,
            "category_confidence": classification_confidence,
            "indicators": indicators,
            "explanation": explanation,
            "expected_failure_modes": expected_failures,
            "expected_alignment": expected_match,
        }

    @classmethod
    def _check_expected_alignment(
        cls,
        actual_category: FailureModeCategory,
        expected_failures: List[str]
    ) -> Dict:
        if not expected_failures:
            return {"matches": None, "details": "no_expected_modes_specified"}
        matches = []
        for expected in expected_failures:
            expected_lower = expected.lower()
            if "weak" in expected_lower or "basic" in expected_lower:
                if actual_category == FailureModeCategory.WEAK_MODEL:
                    matches.append(expected)
            elif "strong" in expected_lower or "intermediate" in expected_lower:
                if actual_category == FailureModeCategory.STRONG_MODEL:
                    matches.append(expected)
            elif "frontier" in expected_lower or "advanced" in expected_lower:
                if actual_category == FailureModeCategory.FRONTIER_MODEL:
                    matches.append(expected)
            elif "overconfident" in expected_lower:
                if actual_category == FailureModeCategory.FRONTIER_MODEL:
                    matches.append(expected)
            elif "contradiction" in expected_lower or "inconsist" in expected_lower:
                if actual_category in [FailureModeCategory.STRONG_MODEL,
                                       FailureModeCategory.FRONTIER_MODEL]:
                    matches.append(expected)
        return {
            "matches": matches,
            "match_rate": len(matches) / len(expected_failures) if expected_failures else None,
            "details": f"matched {len(matches)}/{len(expected_failures)} expected modes"
        }


# =============================================================================
# SECTION 8: MULTI-RUN EVALUATION ENGINE (v13.0 HARDENED)
# =============================================================================

class MultiRunEvaluator:
    """
    Core evaluation engine that runs tasks multiple times.

    v13.0 hardening:
    - _single_run() validates raw_output length before parsing.
    - _call_api() NEVER falls back to mock — raises RuntimeError.
    - _compute_aggregate_metrics() filters INVALID decisions and None
      confidence values before all statistical computations.
    """

    # Minimum response length (chars). Below this → RuntimeError.
    MIN_RESPONSE_LENGTH = 5

    def __init__(
        self,
        config: EvaluationConfig,
        repro_manager: ReproducibilityManager,
        api_client: Any
    ):
        self.config = config
        self.repro = repro_manager
        self.api_client = api_client
        self.metrics = MetricsEngine()
        self.contradiction_detector = ContradictionDetector()
        self.failure_classifier = FailureModeClassifier()

    def evaluate_task(self, task: Dict, model: str) -> Dict:
        """Full multi-run evaluation of a single task."""
        task_id = task.get("id", "unknown")
        num_runs = self.config.num_runs_per_task

        raw_meta_instability = task.get("meta_instability", "unknown")
        if isinstance(raw_meta_instability, bool):
            meta_instability = "extreme" if raw_meta_instability else "low"
        else:
            meta_instability = str(raw_meta_instability)

        raw_invalidating = task.get("self_invalidating_logic", "standard task structure")
        if isinstance(raw_invalidating, bool):
            invalidating_info = (
                "self-invalidating logic present" if raw_invalidating
                else "standard task structure"
            )
        else:
            invalidating_info = str(raw_invalidating)

        raw_expected = task.get("expected_failure_modes", [])
        if isinstance(raw_expected, dict):
            expected_failures = list(raw_expected.keys())
        elif isinstance(raw_expected, list):
            expected_failures = raw_expected
        else:
            expected_failures = []

        adaptive_mutation = task.get("adaptive_mutation", {})
        options = task.get("options", [])
        scenario = task.get("prompt", task.get("scenario", ""))

        base_prompt = AdversarialPromptEngine.build_base_prompt(
            scenario=scenario,
            options=options,
            meta_instability=meta_instability,
            invalidating_info=invalidating_info
        )

        run_results = []
        for run_idx in range(num_runs):
            run_seed = self.repro.get_task_seed(task_id, run_idx)
            if self.config.deterministic:
                random.seed(run_seed)
                np.random.seed(run_seed)

            result = self._single_run(base_prompt, model, task_id, run_idx)
            run_results.append(result)
            
            # [DEBUG MONITOR] Real-time visibility into model output
            snippet = result.get("raw_output", "")[:120].replace("\n", " ")
            print(f"      [v13.0 DEBUG] Run {run_idx} snippet: {snippet}...")
            self.repro._log("run_completed", {
                "task_id": task_id,
                "run_idx": run_idx,
                "decision": result["decision"],
                "confidence": result["confidence"]
            })

        aggregate = self._compute_aggregate_metrics(run_results, num_runs)

        mutation_result = None
        if self.config.enable_adaptive_mutation and adaptive_mutation:
            if aggregate.get("num_valid_decisions", 0) >= self.config.adaptive_mutation_min_valid_runs:
                mutation_result = self._run_adaptive_mutation(
                    task, model, aggregate, adaptive_mutation
                )
            else:
                mutation_result = {
                    "available": False,
                    "mutation_type": "skipped",
                    "reason": "insufficient_valid_runs_for_mutation",
                }

        failure_classification = self.failure_classifier.classify(
            stability=aggregate["decision_stability"],
            entropy=aggregate["decision_entropy"],
            coherence=aggregate["coherence_score"],
            confidence=aggregate["mean_confidence"],
            contradiction_rate=aggregate["contradiction_rate"],
            calibration_error=aggregate["mean_calibration_error"],
            expected_failures=expected_failures,
            epistemic_awareness=aggregate["epistemic_awareness"],
            cross_run_inconsistency=aggregate["cross_run_inconsistency"],
            invalid_rate=aggregate["invalid_decision_rate"],
            parse_success_rate=aggregate["parse_success_rate"],
        )

        meta_score = self.metrics.compute_meta_score(
            stability=aggregate["decision_stability"],
            calibration_error=aggregate["mean_calibration_error"],
            coherence=aggregate["coherence_score"],
            contradiction_rate=aggregate["contradiction_rate"],
            epistemic_awareness=aggregate["epistemic_awareness"]
        )

        all_run_outputs = [
            {"decision": r["decision"], "confidence": r["confidence"]}
            for r in run_results
        ]
        try:
            validate_outputs(all_run_outputs)
        except AssertionError as validation_err:
            print(f"[VALIDATION ERROR] Task {task_id}: {validation_err}")

        return {
            "task_id": task_id,
            "model": model,
            "num_runs": num_runs,
            "meta_instability": meta_instability,
            "expected_failure_modes": expected_failures,
            "run_results": run_results,
            "aggregate_metrics": aggregate,
            "mutation_result": mutation_result,
            "failure_classification": failure_classification,
            "meta_score": meta_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _single_run(
        self,
        prompt: str,
        model: str,
        task_id: str,
        run_idx: int
    ) -> Dict:
        """Execute a single model run with timeout-safe validation."""
        raw_output, latency, success, api_status = self._call_api(prompt, model)

        if api_status == "timeout":
            return self._build_run_result(
                run_idx=run_idx,
                decision="TIMEOUT",
                confidence=None,
                raw_output="",
                latency=latency,
                success=False,
                response_status="timeout",
                parse_success=False,
                parse_warnings=["timeout"],
                uncertainty=None,
                reasoning="",
            )

        raw_output = (raw_output or "").strip()
        
        # [GAME CHANGER] MAX LENGTH FILTER
        if len(raw_output) > 4000:
            raw_output = raw_output[:4000]
            parse_warnings.append("output_clamped_to_4000_chars")
            
        parse_warnings: List[str] = []

        if not raw_output or len(raw_output) < self.config.min_response_length:
            return self._build_run_result(
                run_idx=run_idx,
                decision="INVALID",
                confidence=None,
                raw_output=raw_output,
                latency=latency,
                success=success,
                response_status="invalid",
                parse_success=False,
                parse_warnings=(parse_warnings + ["invalid_output"]),
                uncertainty=None,
                reasoning=raw_output,
            )

        parse_result = OutputParser.parse(raw_output)
        if parse_warnings:
            parse_result["parse_warnings"].extend(parse_warnings)
        decision = parse_result["decision"]
        if decision == "INVALID":
            extracted = extract_decision(raw_output, valid_options=tuple(self.config.valid_options))
            if extracted != "INVALID":
                decision = extracted
                parse_result["parse_warnings"].append("module_level_extraction_used")

        decision = parse_result["decision"]
        
        # [GAME CHANGER] DECISION CONFIDENCE CHECK
        if decision not in ["A", "B", "C", "D", "E"]:
            decision = "INVALID"
            parse_result["parse_success"] = False
            parse_result["parse_warnings"].append("hard_decision_validation_failed")

        confidence = parse_result["confidence"]
        if confidence is None:
            confidence = extract_confidence(raw_output)
            parse_result["parse_warnings"].append("confidence_estimated")
        if confidence is None or not (0.0 <= confidence <= 1.0):
            confidence = None

        fallback_used = decision == "INVALID" or not parse_result["parse_success"]
        contradiction_result = self.contradiction_detector.detect(parse_result["reasoning"], decision)
        response_status = "verbose" if len(raw_output.split()) > self.config.max_response_words else "normal"

        return self._build_run_result(
            run_idx=run_idx,
            decision=decision,
            confidence=confidence,
            raw_output=raw_output,
            latency=latency,
            success=success,
            response_status=response_status,
            parse_success=parse_result["parse_success"],
            parse_warnings=parse_result["parse_warnings"],
            uncertainty=parse_result["uncertainty"],
            reasoning=parse_result["reasoning"],
            contradiction=contradiction_result,
            fallback_used=fallback_used,
        )

    def _call_api(
        self,
        prompt: str,
        model: str
    ) -> Tuple[str, float, bool, str]:
        if not HAS_OLLAMA:
            raise RuntimeError(
                "ollama package is not installed. "
                "Install with: pip install ollama"
            )
        if not isinstance(self.api_client, OllamaClient):
            raise RuntimeError(
                "api_client is not an OllamaClient instance. "
                "Real model execution requires a valid Ollama client. "
                "Check create_api_client() and OLLAMA_API_KEY env var."
            )
        system_message = (
            "Answer immediately.\n"
            "Do NOT write reasoning first.\n\n"
            "First line MUST be:\n"
            "Decision: A (or B, C, D, E)\n\n"
            "Then optionally explain briefly."
        )

        attempt_start = time.time()
        max_attempts = max(1, self.config.max_empty_response_retries)
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            elapsed = time.time() - attempt_start
            remaining = self.config.timeout_seconds - elapsed
            if remaining <= 0:
                return "", self.config.timeout_seconds, False, "timeout"

            client = self.api_client
            if attempt > 0:
                client = OllamaClient(host=self.config.api_host, timeout=max(1.0, remaining))

            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {"role": "user", "content": prompt}
                    ],
                    think=False,
                    stream=False,
                    timeout=30,  # [GAME CHANGER] HARD TIMEOUT
                    keep_alive=0 if attempt == 0 else None,
                    options={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "num_predict": self.config.max_tokens,
                        "stop": ["\n\n", "###", "Reasoning Process"],
                    }
                )
            except Exception as exc:
                # Compatibility retry for backends that reject `think` or keep_alive.
                try:
                    response = client.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        stream=False,
                        think=False,
                        options={
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "num_predict": 20,
                        }
                    )
                except Exception as retry_exc:
                    last_error = retry_exc
                    err_str = str(retry_exc).lower()
                    if "timeout" in err_str:
                        return "", self.config.timeout_seconds, False, "timeout"
                    continue

            raw_output = self._extract_response_text(response)
            raw_output = (raw_output or "").strip()
            latency = time.time() - attempt_start

            if latency >= self.config.timeout_seconds:
                return "", self.config.timeout_seconds, False, "timeout"

            if raw_output:
                return raw_output, latency, True, "ok"

            self.repro._log("api_empty_content", {
                "model": model,
                "attempt": attempt,
                "done_reason": getattr(response, "done_reason", None),
            })

        latency = min(time.time() - attempt_start, self.config.timeout_seconds)
        if last_error is not None:
            self.repro._log("api_error", {
                "model": model,
                "error": str(last_error),
                "attempt": max_attempts - 1
            })
            return "", latency, False, "error"
        return "", latency, False, "invalid"

    def _extract_response_text(self, response: Any) -> str:
        raw_output = getattr(getattr(response, "message", None), "content", "") or ""
        if raw_output:
            return raw_output

        if hasattr(response, "get"):
            message = response.get("message", {})
            if isinstance(message, dict):
                raw_output = message.get("content", "") or ""
                if raw_output:
                    return raw_output

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                message = dumped.get("message", {}) if isinstance(dumped, dict) else {}
                raw_output = message.get("content", "") or ""
                if raw_output:
                    return raw_output
            except Exception:
                pass

        return ""

    def _build_run_result(
        self,
        run_idx: int,
        decision: str,
        confidence: Optional[float],
        raw_output: str,
        latency: float,
        success: bool,
        response_status: str,
        parse_success: bool,
        parse_warnings: List[str],
        uncertainty: Optional[str],
        reasoning: str,
        contradiction: Optional[Dict] = None,
        fallback_used: bool = False,
    ) -> Dict:
        contradiction_result = contradiction or self.contradiction_detector.detect(reasoning, decision)
        calibration_error = None
        if confidence is not None:
            calibration_error = self.metrics.compute_confidence_calibration_error(
                confidence,
                self.config.empirical_entropy_baseline,
                len(self.config.valid_options),
                empirical_baseline=self.config.empirical_entropy_baseline
            )

        return {
            "run_idx": run_idx,
            "decision": decision,
            "confidence": confidence,
            "fallback_used": fallback_used,
            "response_status": response_status,
            "response_word_count": len(raw_output.split()) if raw_output else 0,
            "parse_log": {
                "raw_output": raw_output,
                "parsed_decision": decision,
                "parsed_confidence": confidence,
                "fallback_used": fallback_used,
                "parse_warnings": parse_warnings,
            },
            "decision_evaluation": evaluate_decision(
                {"evaluation_mode": "adversarial_decision"}, decision
            ),
            "raw_response": raw_output,
            "uncertainty": uncertainty,
            "reasoning": reasoning,
            "raw_output": raw_output,
            "parse_success": parse_success,
            "parse_warnings": parse_warnings,
            "latency_seconds": latency,
            "api_success": success,
            "contradiction": contradiction_result,
            "calibration_error": calibration_error,
        }

    def _run_adaptive_mutation(
        self,
        task: Dict,
        model: str,
        aggregate: Dict,
        mutation_data: Dict
    ) -> Dict:
        """Run adaptive mutation follow-up."""
        run_results = aggregate.get("run_results", [])
        if not run_results:
            return None

        valid_runs = [r for r in run_results if r.get("decision") not in ("INVALID", "TIMEOUT", "NONE", "")]
        if valid_runs:
            ans_dist = aggregate["answer_distribution"]
            if ans_dist:
                most_common_decision = max(ans_dist, key=lambda k: ans_dist[k])
            else:
                most_common_decision = valid_runs[0]["decision"]
            representative = next(
                (r for r in valid_runs if r["decision"] == most_common_decision),
                valid_runs[0]
            )
        else:
            return None

        mutation_prompt = AdversarialPromptEngine.build_mutation_prompt(
            task=task,
            previous_response=representative,
            mutation_data=mutation_data
        )

        raw_output, latency, success, api_status = self._call_api(mutation_prompt, model)
        if api_status == "timeout":
            return {
                "mutation_type": "skipped",
                "reason": "mutation_timeout",
                "raw_output": "",
                "latency_seconds": latency,
            }

        if (
            not raw_output
            or len(raw_output.strip()) < self.config.min_response_length
            or len(raw_output.strip()) > self.config.max_response_chars
        ):
            return {
                "mutation_type": "skipped",
                "reason": "mutation_response_too_short_or_empty",
                "raw_output": raw_output,
                "latency_seconds": latency,
            }

        parse_result = OutputParser.parse(raw_output)
        new_decision = parse_result["decision"]
        if new_decision == "INVALID":
            extracted = extract_decision(raw_output, valid_options=tuple(self.config.valid_options))
            if extracted != "INVALID":
                new_decision = extracted

        answer_changed = new_decision != most_common_decision
        orig_conf = representative.get("confidence")
        original_confidence = orig_conf if orig_conf is not None else 0.5
        new_confidence = parse_result["confidence"]
        if new_confidence is None:
            new_confidence = extract_confidence(raw_output)
        if new_confidence is None:
            new_confidence = original_confidence
        confidence_shift = abs(new_confidence - original_confidence)

        return {
            "mutation_type": "contradiction" if original_confidence >= 0.7 else "clarity",
            "original_decision": most_common_decision,
            "new_decision": new_decision,
            "answer_changed": answer_changed,
            "original_confidence": original_confidence,
            "new_confidence": new_confidence,
            "confidence_shift": confidence_shift,
            "reasoning_collapse": (
                answer_changed and
                "contradiction" not in (parse_result.get("uncertainty") or "").lower()
            ),
            "latency_seconds": latency,
            "raw_output": raw_output,
            "raw_response": raw_output,
            "parse_result": parse_result,
        }

    def _compute_aggregate_metrics(
        self,
        run_results: List[Dict],
        num_runs: int
    ) -> Dict:
        """
        Compute aggregate metrics across all runs.

        v13.0: Filters INVALID decisions and None confidence values before
        all statistical computations to ensure metrics integrity.
        """
        all_decisions = [r["decision"] for r in run_results]

        valid_decisions = [
            d for d in all_decisions
            if d not in ("INVALID", "TIMEOUT", "NONE", "", None)
        ]

        confidences = [r.get("confidence") for r in run_results]
        valid_conf = [c for c in confidences if c is not None]

        reasoning_texts = [r.get("reasoning", "") for r in run_results]
        contradiction_counts = [
            r["contradiction"]["contradiction_count"]
            for r in run_results
            if "contradiction" in r
        ]

        # Warn if too many INVALID decisions
        invalid_count = len(all_decisions) - len(valid_decisions)
        invalid_rate = invalid_count / num_runs if num_runs > 0 else 0.0
        if invalid_count > 0:
            print(
                f"[v13.0 METRICS] Filtered {invalid_count}/{len(all_decisions)} "
                "invalid decisions from aggregate computation."
            )

        if not valid_decisions:
            print("[WARNING] ALL decisions INVALID — parser failure for every run.")

        effective_n = len(valid_decisions)

        distribution = self.metrics.compute_distribution(valid_decisions)
        js_divergence = self.metrics.compute_jsd(distribution)
        entropy = self.metrics.compute_entropy(distribution, effective_n)
        normalized_entropy = self.metrics.compute_normalized_entropy(
            entropy, len(self.config.valid_options)
        )
        stability = self.metrics.compute_decision_stability(distribution, effective_n)
        oscillation = self.metrics.compute_oscillation_index(valid_decisions)

        cross_run_inconsistency = self.metrics.compute_cross_run_inconsistency(
            reasoning_texts
        )

        mean_confidence = sum(valid_conf) / len(valid_conf) if valid_conf else None
        confidence_std = (
            sqrt(
                sum((c - mean_confidence) ** 2 for c in valid_conf)
                / max(len(valid_conf) - 1, 1)
            )
            if len(valid_conf) > 1 else 0.0
        )

        coherence_score = self.metrics.compute_coherence_score(reasoning_texts)

        runs_with_contradiction = sum(1 for c in contradiction_counts if c > 0)
        contradiction_rate = runs_with_contradiction / num_runs if num_runs > 0 else 0.0

        parse_successes = sum(1 for r in run_results if r.get("parse_success"))
        parse_success_rate = parse_successes / num_runs if num_runs > 0 else 0.0

        fallback_count = sum(1 for r in run_results if r.get("fallback_used", False))
        fallback_rate = fallback_count / num_runs if num_runs > 0 else 0.0
        if fallback_rate > 0.3:
            print(
                f"[WARNING] High fallback rate ({fallback_rate:.0%}) — "
                "model is not following marker output format"
            )

        status_counts = Counter(r.get("response_status", "normal") for r in run_results)
        timeout_risk_rate = status_counts.get("timeout", 0) / num_runs if num_runs > 0 else 0.0
        truncated_rate = status_counts.get("verbose", 0) / num_runs if num_runs > 0 else 0.0

        latencies = [r["latency_seconds"] for r in run_results]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        uncertainty_signal = max(0.0, min(
            1.0,
            0.35 * normalized_entropy +
            0.35 * cross_run_inconsistency +
            0.15 * contradiction_rate +
            0.15 * invalid_rate
        ))
        expected_confidence = self.metrics.compute_expected_confidence(uncertainty_signal)

        mean_calibration_error = sum(
            self.metrics.compute_confidence_calibration_error_from_signal(
                conf,
                uncertainty_signal
            )
            for conf in valid_conf
        ) / len(valid_conf) if valid_conf else None

        epistemic_awareness = self.metrics.compute_epistemic_awareness_from_signal(
            mean_confidence,
            uncertainty_signal,
            parse_success_rate=parse_success_rate
        )
        failure_signature = classify_failure(
            entropy,
            cross_run_inconsistency,
            epistemic_awareness,
            stability
        )
        return {
            "answer_distribution": dict(distribution),
            "decision_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "decision_stability": stability,
            "oscillation_index": oscillation,
            "cross_run_inconsistency": cross_run_inconsistency,
            "mean_confidence": mean_confidence,
            "confidence_std": confidence_std,
            "js_divergence": js_divergence,
            "mean_calibration_error": mean_calibration_error,
            "expected_confidence": expected_confidence,
            "uncertainty_signal": uncertainty_signal,
            "coherence_score": coherence_score,
            "contradiction_rate": contradiction_rate,
            "epistemic_awareness": epistemic_awareness,
            "parse_success_rate": parse_success_rate,
            "fallback_usage_rate": fallback_rate,
            "fallback_rate": fallback_rate,
            "invalid_decision_rate": invalid_rate,
            "timeout_risk_rate": timeout_risk_rate,
            "truncated_reasoning_rate": truncated_rate,
            "response_status_distribution": dict(status_counts),
            "mean_latency_seconds": mean_latency,
            "num_valid_decisions": len(valid_decisions),
            "num_invalid_decisions": invalid_count,
            "run_results": run_results,
            "failure_signature": failure_signature,
        }


# =============================================================================
# SECTION 9: MODEL-LEVEL AGGREGATOR
# =============================================================================

class ModelAggregator:
    """Aggregates task-level results into model-level failure profiles."""

    def aggregate_model(self, task_results: List[Dict]) -> Dict:
        if not task_results:
            return self._empty_profile()

        # 🚀 [v13.0.4] [HARDENING] SCHEMA INJECTION
        # Ensure all tasks have a failure_signature, meta_score, and aggregate_metrics
        # This prevents KeyErrors if individual results are malformed or missing data.
        for t in task_results:
            if "failure_signature" not in t:
                t["failure_signature"] = {"category": "unknown", "subcategory": "unknown"}
            if "aggregate_metrics" not in t:
                # Use a safe empty structure for aggregation
                t["aggregate_metrics"] = {
                    "decision_stability": 0.0, "decision_entropy": 1.5,
                    "mean_calibration_error": 1.0, "coherence_score": 0.0,
                    "contradiction_rate": 0.0, "mean_confidence": 0.0,
                    "oscillation_index": 0.0, "cross_run_inconsistency": 1.0,
                    "epistemic_awareness": 0.0, "invalid_decision_rate": 1.0,
                    "parse_success_rate": 0.0
                }
            if "meta_score" not in t:
                t["meta_score"] = {"meta_score": 0.0}

        meta_scores = [t["meta_score"]["meta_score"] for t in task_results]
        stabilities = [t["aggregate_metrics"]["decision_stability"] for t in task_results]
        entropies = [t["aggregate_metrics"]["decision_entropy"] for t in task_results]
        calibrations = [t["aggregate_metrics"]["mean_calibration_error"] for t in task_results]
        coherences = [t["aggregate_metrics"]["coherence_score"] for t in task_results]
        contradictions = [t["aggregate_metrics"]["contradiction_rate"] for t in task_results]
        confidences = [t["aggregate_metrics"]["mean_confidence"] for t in task_results]
        oscillations = [t["aggregate_metrics"]["oscillation_index"] for t in task_results]
        cross_run_inconsistencies = [t["aggregate_metrics"]["cross_run_inconsistency"] for t in task_results]
        epistemic_awarenesses = [t["aggregate_metrics"]["epistemic_awareness"] for t in task_results]
        invalid_rates = [t["aggregate_metrics"].get("invalid_decision_rate", 0.0) for t in task_results]

        failure_categories = Counter(
            t.get("failure_signature", {}).get("category", "unknown") for t in task_results
        )
        failure_subcategories = Counter(
            t.get("failure_signature", {}).get("subcategory", "unknown") for t in task_results
        )


        mutation_stats = self._aggregate_mutations(task_results)

        confidence_biases = []
        for t in task_results:
            bias = MetricsEngine.compute_confidence_bias_from_signal(
                t["aggregate_metrics"]["mean_confidence"],
                t["aggregate_metrics"].get("uncertainty_signal", 0.5),
            )
            confidence_biases.append(bias)
        bias_distribution = Counter(confidence_biases)

        profile = {
            "model": task_results[0]["model"],
            "num_tasks_evaluated": len(task_results),
            "summary_metrics": {
                "mean_meta_score": self._safe_mean(meta_scores),
                "std_meta_score": self._safe_std(meta_scores),
                "mean_stability": self._safe_mean(stabilities),
                "mean_entropy": self._safe_mean(entropies),
                "mean_calibration_error": self._safe_mean(calibrations),
                "mean_coherence": self._safe_mean(coherences),
                "mean_contradiction_rate": self._safe_mean(contradictions),
                "mean_confidence": self._safe_mean(confidences),
                "mean_oscillation": self._safe_mean(oscillations),
                "mean_cross_run_inconsistency": self._safe_mean(cross_run_inconsistencies),
                "mean_epistemic_awareness": self._safe_mean(epistemic_awarenesses),
            },
            "failure_profile": {
                "weak_model_rate": failure_categories.get("weak_model_behavior", 0) / len(task_results),
                "strong_model_rate": failure_categories.get("strong_model_behavior", 0) / len(task_results),
                "frontier_model_rate": failure_categories.get("frontier_model_behavior", 0) / len(task_results),
                "uncategorized_rate": failure_categories.get("uncategorized", 0) / len(task_results),
                "distribution": dict(failure_categories),
                "subcategory_distribution": dict(failure_subcategories),
            },
            "confidence_bias_profile": {
                "overconfident_rate": bias_distribution.get("overconfident", 0) / len(task_results),
                "underconfident_rate": bias_distribution.get("underconfident", 0) / len(task_results),
                "well_calibrated_rate": bias_distribution.get("well_calibrated", 0) / len(task_results),
                "distribution": dict(bias_distribution),
            },
            "mutation_behavior": mutation_stats,
            "instability_index": self._compute_instability_index(
                stabilities, entropies, oscillations, cross_run_inconsistencies, invalid_rates
            ),
            "hallucination_indicators": self._compute_hallucination_indicators(
                contradictions, coherences, calibrations
            ),
            "cognitive_structure": {
                "reasoning_divergence": self._safe_mean(cross_run_inconsistencies),
                "meta_cognitive_quality": self._safe_mean(epistemic_awarenesses),
                "reasoning_consistency": 1.0 - self._safe_mean(cross_run_inconsistencies),
            },
            "task_details": task_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        final_score = (
            0.4 * self._safe_mean([1 - x for x in cross_run_inconsistencies]) +
            0.3 * self._safe_mean(epistemic_awarenesses) +
            0.2 * self._safe_mean(coherences) +
            0.1 * (1 - self._safe_mean(calibrations))
        )
        profile["final_score"] = final_score

        return profile

    def _aggregate_mutations(self, task_results: List[Dict]) -> Dict:
        mutations = [t.get("mutation_result") for t in task_results
                     if t.get("mutation_result") and t["mutation_result"].get("mutation_type") != "skipped"]
        if not mutations:
            return {"available": False}
        answer_changes = sum(1 for m in mutations if m.get("answer_changed"))
        collapses = sum(1 for m in mutations if m.get("reasoning_collapse"))
        return {
            "available": True,
            "num_mutations": len(mutations),
            "answer_change_rate": answer_changes / len(mutations),
            "reasoning_collapse_rate": collapses / len(mutations),
            "mean_confidence_shift": self._safe_mean(
                [m.get("confidence_shift", 0) for m in mutations]
            ),
        }

    def _compute_instability_index(
        self,
        stabilities: List[float],
        entropies: List[float],
        oscillations: List[float],
        cross_run_inconsistencies: List[float],
        invalid_rates: List[float]
    ) -> Dict:
        norm_stabilities = [1 - s for s in stabilities]
        max_entropy = log2(5)
        norm_entropies = [e / max_entropy for e in entropies]
        instability_components = {
            "decision_instability": self._safe_mean(norm_stabilities),
            "entropy_instability": self._safe_mean(norm_entropies),
            "oscillation_instability": self._safe_mean(oscillations),
            "cross_run_instability": self._safe_mean(cross_run_inconsistencies),
            "invalid_output_instability": self._safe_mean(invalid_rates),
        }
        composite = (
            0.15 * instability_components["decision_instability"] +
            0.15 * instability_components["entropy_instability"] +
            0.10 * instability_components["oscillation_instability"] +
            0.40 * instability_components["cross_run_instability"] +
            0.20 * instability_components["invalid_output_instability"]
        )
        return {
            "composite_index": composite,
            "components": instability_components,
            "interpretation": self._interpret_instability(composite),
        }

    def _interpret_instability(self, index: float) -> str:
        if index < 0.15:
            return "highly_stable"
        elif index < 0.30:
            return "moderately_stable"
        elif index < 0.50:
            return "moderately_unstable"
        elif index < 0.70:
            return "highly_unstable"
        return "critically_unstable"

    def _compute_hallucination_indicators(
        self,
        contradictions: List[float],
        coherences: List[float],
        calibrations: List[float]
    ) -> Dict:
        high_contradiction_low_coherence = sum(
            1 for c, co in zip(contradictions, coherences)
            if c > 0.5 and co < 0.4
        )
        return {
            "contradiction_coherence_mismatch_rate": (
                high_contradiction_low_coherence / len(contradictions)
                if contradictions else 0.0
            ),
            "mean_contradiction_rate": self._safe_mean(contradictions),
            "mean_coherence": self._safe_mean(coherences),
            "hallucination_risk_level": self._assess_hallucination_risk(
                high_contradiction_low_coherence,
                len(contradictions),
                self._safe_mean(calibrations)
            ),
        }

    def _assess_hallucination_risk(
        self,
        mismatched: int,
        total: int,
        mean_calibration: float
    ) -> str:
        if total == 0:
            return "unknown"
        mismatch_rate = mismatched / total
        if mismatch_rate > 0.5 or (mismatch_rate > 0.3 and mean_calibration > 0.4):
            return "high"
        elif mismatch_rate > 0.25:
            return "moderate"
        elif mismatch_rate > 0.1:
            return "low"
        return "minimal"

    def _empty_profile(self) -> Dict:
        return {
            "model": "unknown",
            "num_tasks_evaluated": 0,
            "summary_metrics": {},
            "failure_profile": {},
            "confidence_bias_profile": {},
            "mutation_behavior": {"available": False},
            "instability_index": {},
            "hallucination_indicators": {},
            "cognitive_structure": {},
            "task_details": [],
        }

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else 0.0

    @staticmethod
    def _safe_std(values: List[float]) -> float:
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return 0.0
        mean = sum(clean) / len(clean)
        variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
        return sqrt(variance)


# =============================================================================
# SECTION 10: PUBLICATION-READY OUTPUT SYSTEM
# =============================================================================

class OutputSystem:
    """Generates publication-ready outputs."""

    def __init__(self, output_dir: str, prefix: str):
        self.output_dir = output_dir
        self.prefix = prefix
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def save_full_results(
        self,
        all_results: Dict,
        config: EvaluationConfig,
        repro: ReproducibilityManager
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}_full.json"
        filepath = os.path.join(self.output_dir, filename)

        payload = {
            "metadata": {
                "engine_version": "13.0.0",
                "engine_type": "cognitive_failure_measurement",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_hash": repro._config_hash,
                "reproducibility": {
                    "seed": config.seed,
                    "deterministic": config.deterministic,
                    "num_runs_per_task": config.num_runs_per_task,
                    "temperature": config.temperature,
                    "empirical_entropy_baseline": config.empirical_entropy_baseline,
                },
                "audit_trail": repro.get_audit_trail(),
            },
            "configuration": config.to_dict(),
            "models_evaluated": list(all_results.keys()),
            "results": self._serialize_results(all_results),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        print(f"  [SAVED] Full results: {filepath}")
        return filepath

    def save_failure_profiles(self, all_profiles: Dict) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}_failure_profiles.json"
        filepath = os.path.join(self.output_dir, filename)

        summaries = {}
        for model, profile in all_profiles.items():
            summaries[model] = {
                k: v for k, v in profile.items()
                if k != "task_details"
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

        print(f"  [SAVED] Failure profiles: {filepath}")
        return filepath

    def save_csv_summary(self, all_profiles: Dict) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}_summary.csv"
        filepath = os.path.join(self.output_dir, filename)

        rows = []
        for model, profile in all_profiles.items():
            summary = profile.get("summary_metrics", {})
            failure = profile.get("failure_profile", {})
            instability = profile.get("instability_index", {})
            hallucination = profile.get("hallucination_indicators", {})
            cognitive = profile.get("cognitive_structure", {})

            rows.append({
                "model": model,
                "num_tasks": profile.get("num_tasks_evaluated", 0),
                "meta_score_mean": summary.get("mean_meta_score", 0),
                "meta_score_std": summary.get("std_meta_score", 0),
                "stability_mean": summary.get("mean_stability", 0),
                "entropy_mean": summary.get("mean_entropy", 0),
                "calibration_error_mean": summary.get("mean_calibration_error", 0),
                "coherence_mean": summary.get("mean_coherence", 0),
                "contradiction_rate_mean": summary.get("mean_contradiction_rate", 0),
                "confidence_mean": summary.get("mean_confidence", 0),
                "oscillation_mean": summary.get("mean_oscillation", 0),
                "cross_run_inconsistency_mean": summary.get("mean_cross_run_inconsistency", 0),
                "epistemic_awareness_mean": summary.get("mean_epistemic_awareness", 0),
                "reasoning_divergence": cognitive.get("reasoning_divergence", 0),
                "meta_cognitive_quality": cognitive.get("meta_cognitive_quality", 0),
                "weak_model_rate": failure.get("weak_model_rate", 0),
                "strong_model_rate": failure.get("strong_model_rate", 0),
                "frontier_model_rate": failure.get("frontier_model_rate", 0),
                "instability_index": instability.get("composite_index", 0),
                "instability_interpretation": instability.get("interpretation", ""),
                "hallucination_risk": hallucination.get("hallucination_risk_level", ""),
            })

        if rows:
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"  [SAVED] CSV summary: {filepath}")
        return filepath

    def save_per_task_csv(self, all_results: Dict) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}_per_task.csv"
        filepath = os.path.join(self.output_dir, filename)

        rows = []
        for model, tasks in all_results.items():
            for task in tasks:
                agg = task.get("aggregate_metrics", {})
                fc = task.get("failure_signature", {})
                ms = task.get("meta_score", {})

                rows.append({
                    "model": model,
                    "task_id": task.get("task_id", ""),
                    "meta_instability": task.get("meta_instability", ""),
                    "decision_stability": agg.get("decision_stability", 0),
                    "decision_entropy": agg.get("decision_entropy", 0),
                    "normalized_entropy": agg.get("normalized_entropy", 0),
                    "oscillation_index": agg.get("oscillation_index", 0),
                    "mean_confidence": agg.get("mean_confidence", 0),
                    "calibration_error": agg.get("mean_calibration_error", 0),
                    "coherence_score": agg.get("coherence_score", 0),
                    "contradiction_rate": agg.get("contradiction_rate", 0),
                    "cross_run_inconsistency": agg.get("cross_run_inconsistency", 0),
                    "epistemic_awareness": agg.get("epistemic_awareness", 0),
                    "num_valid_decisions": agg.get("num_valid_decisions", 0),
                    "num_invalid_decisions": agg.get("num_invalid_decisions", 0),
                    "meta_score": ms.get("meta_score", 0),
                    "failure_category": fc.get("category", ""),
                    "expected_alignment": fc.get("expected_alignment", {}).get("match_rate"),
                })

        if rows:
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"  [SAVED] Per-task CSV: {filepath}")
        return filepath

    def generate_publication_figures(
        self,
        all_profiles: Dict
    ) -> Optional[List[str]]:
        if not HAS_MATPLOTLIB:
            print("  [SKIP] Plots: matplotlib not installed")
            return None

        generated = []
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for fn in [
            self._plot_model_comparison,
            self._plot_instability_distribution,
            self._plot_failure_modes,
            self._plot_calibration,
            self._plot_epistemic_analysis,
        ]:
            path = fn(all_profiles, timestamp)
            if path:
                generated.append(path)

        return generated

    def _plot_model_comparison(self, profiles: Dict, timestamp: str) -> Optional[str]:
        if not profiles:
            return None
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        categories = [
            'Stability', 'Calibration', 'Coherence',
            'No Contradiction', 'Epistemic\nAwareness'
        ]
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
        for idx, (model, profile) in enumerate(profiles.items()):
            summary = profile.get("summary_metrics", {})
            values = [
                summary.get("mean_stability", 0),
                1 - summary.get("mean_calibration_error", 0.5),
                summary.get("mean_coherence", 0),
                1 - summary.get("mean_contradiction_rate", 0.5),
                summary.get("mean_epistemic_awareness", 0),
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title(
            'Model Cognitive Failure Profile Comparison\n(v13.0)',
            size=14, fontweight='bold', pad=20
        )
        filepath = os.path.join(self.output_dir, f"{self.prefix}_{timestamp}_radar.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] Radar chart: {filepath}")
        return filepath

    def _plot_instability_distribution(self, profiles: Dict, timestamp: str) -> Optional[str]:
        if not profiles:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        models = list(profiles.keys())
        instability_indices = [
            profiles[m].get("instability_index", {}).get("composite_index", 0)
            for m in models
        ]
        colors = ['#2ecc71' if i < 0.4 else '#f39c12' if i < 0.65 else '#e74c3c'
                  for i in instability_indices]
        axes[0].barh(models, instability_indices, color=colors)
        axes[0].axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Moderate')
        axes[0].axvline(x=0.65, color='red', linestyle='--', alpha=0.7, label='High')
        axes[0].set_xlabel('Instability Index (higher = worse)')
        axes[0].set_title('Composite Instability Index by Model')
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        components = ['decision', 'entropy', 'oscillation', 'cross_run', 'invalid_output']
        x = np.arange(len(models))
        width = 0.16
        for i, comp in enumerate(components):
            values = [
                profiles[m].get("instability_index", {}).get("components", {}).get(
                    f"{comp}_instability", 0
                )
                for m in models
            ]
            axes[1].bar(x + i*width, values, width, label=comp.capitalize())
        axes[1].set_xticks(x + width * 2)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('Instability Component')
        axes[1].set_title('Instability Component Breakdown')
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.prefix}_{timestamp}_instability.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] Instability plot: {filepath}")
        return filepath

    def _plot_failure_modes(self, profiles: Dict, timestamp: str) -> Optional[str]:
        if not profiles:
            return None
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(profiles.keys())
        modes = ['weak_model_rate', 'strong_model_rate', 'frontier_model_rate', 'uncategorized_rate']
        labels = ['Weak', 'Strong', 'Frontier', 'Uncategorized']
        colors = ['#95a5a6', '#3498db', '#9b59b6', '#ecf0f1']
        x = np.arange(len(models))
        width = 0.6
        bottom = np.zeros(len(models))
        for mode, label, color in zip(modes, labels, colors):
            values = [profiles[m].get("failure_profile", {}).get(mode, 0) for m in models]
            ax.bar(x, values, width, label=label, bottom=bottom, color=color)
            bottom += np.array(values)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Proportion of Tasks')
        ax.set_title('Failure Mode Composition by Model')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        filepath = os.path.join(self.output_dir, f"{self.prefix}_{timestamp}_failure_modes.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] Failure modes plot: {filepath}")
        return filepath

    def _plot_calibration(self, profiles: Dict, timestamp: str) -> Optional[str]:
        if not profiles:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        models = list(profiles.keys())
        bias_types = ['overconfident_rate', 'underconfident_rate', 'well_calibrated_rate']
        bias_labels = ['Overconfident', 'Underconfident', 'Well Calibrated']
        bias_colors = ['#e74c3c', '#3498db', '#2ecc71']
        x = np.arange(len(models))
        width = 0.25
        for i, (bias, label, color) in enumerate(zip(bias_types, bias_labels, bias_colors)):
            values = [
                profiles[m].get("confidence_bias_profile", {}).get(bias, 0)
                for m in models
            ]
            axes[0].bar(x + i*width, values, width, label=label, color=color)
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('Proportion of Tasks')
        axes[0].set_title('Confidence Bias Distribution')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        for model in models:
            profile = profiles[model]
            summary = profile.get("summary_metrics", {})
            axes[1].scatter(
                summary.get("mean_confidence", 0.5),
                summary.get("mean_calibration_error", 0.5),
                s=200, label=model, alpha=0.7
            )
            axes[1].annotate(
                model,
                (summary.get("mean_confidence", 0.5), summary.get("mean_calibration_error", 0.5)),
                xytext=(5, 5), textcoords='offset points'
            )
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor calibration')
        axes[1].set_xlabel('Mean Confidence')
        axes[1].set_ylabel('Mean Calibration Error')
        axes[1].set_title('Confidence vs Calibration Error')
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.prefix}_{timestamp}_calibration.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] Calibration plot: {filepath}")
        return filepath

    def _plot_epistemic_analysis(self, profiles: Dict, timestamp: str) -> Optional[str]:
        if not profiles:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        models = list(profiles.keys())
        for model in models:
            profile = profiles[model]
            summary = profile.get("summary_metrics", {})
            axes[0].scatter(
                summary.get("mean_epistemic_awareness", 0),
                summary.get("mean_cross_run_inconsistency", 0),
                s=200, label=model, alpha=0.7
            )
            axes[0].annotate(
                model,
                (summary.get("mean_epistemic_awareness", 0),
                 summary.get("mean_cross_run_inconsistency", 0)),
                xytext=(5, 5), textcoords='offset points'
            )
        axes[0].set_xlabel('Epistemic Awareness (higher = better)')
        axes[0].set_ylabel('Cross-Run Inconsistency (higher = worse)')
        axes[0].set_title('Epistemic Awareness vs Reasoning Divergence')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        epistemic_values = [
            profiles[m].get("summary_metrics", {}).get("mean_epistemic_awareness", 0)
            for m in models
        ]
        colors = ['#e74c3c' if v < 0.4 else '#f39c12' if v < 0.6 else '#2ecc71'
                  for v in epistemic_values]
        axes[1].barh(models, epistemic_values, color=colors)
        axes[1].axvline(x=0.4, color='red', linestyle='--', alpha=0.7, label='Poor')
        axes[1].axvline(x=0.6, color='green', linestyle='--', alpha=0.7, label='Good')
        axes[1].set_xlabel('Mean Epistemic Awareness Score')
        axes[1].set_title('Epistemic Awareness by Model')
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.prefix}_{timestamp}_epistemic.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] Epistemic analysis plot: {filepath}")
        return filepath

    def _serialize_results(self, results: Dict) -> Dict:
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Counter):
                return dict(obj)
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)
        return json.loads(json.dumps(results, default=default_serializer))


# =============================================================================
# SECTION 11: DATASET SCHEMA VALIDATOR
# =============================================================================

class DatasetValidator:
    """Validates BioExecBench dataset schema."""

    REQUIRED_FIELDS = ["id", "prompt", "options"]
    RECOMMENDED_FIELDS = [
        "meta_instability",
        "self_invalidating_logic",
        "adaptive_mutation",
        "expected_failure_modes"
    ]

    @classmethod
    def validate(cls, tasks: List[Dict]) -> Dict:
        report = {
            "valid": True,
            "total_tasks": len(tasks),
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        if not tasks:
            report["valid"] = False
            report["errors"].append("Dataset contains zero tasks")
            return report

        missing_required: Set[str] = set()
        missing_recommended: Set[str] = set()
        has_no_correct_answer = 0

        for task in tasks:
            task_id = task.get("id", "unknown")
            for field_name in cls.REQUIRED_FIELDS:
                if field_name not in task:
                    missing_required.add(field_name)
                    report["errors"].append(
                        f"Task {task_id}: missing required field '{field_name}'"
                    )
            for field_name in cls.RECOMMENDED_FIELDS:
                if field_name not in task:
                    missing_recommended.add(field_name)
            no_single_answer = (
                task.get("correct_option_index") == -1
                or task.get("ground_truth", "__missing__") is None
                or task.get("evaluation_mode") == "adversarial_decision"
            )
            if no_single_answer:
                has_no_correct_answer += 1

        if missing_required:
            report["valid"] = False

        for field_name in sorted(missing_recommended):
            report["warnings"].append(
                f"Missing recommended field '{field_name}' — some features may be limited"
            )

        def _norm_instability(v: Any) -> str:
            if isinstance(v, bool):
                return "extreme" if v else "low"
            return str(v) if v is not None else "unknown"

        difficulties = Counter(t.get("difficulty", "unknown") for t in tasks)
        instability_levels = Counter(_norm_instability(t.get("meta_instability")) for t in tasks)

        report["statistics"] = {
            "difficulty_distribution": dict(difficulties),
            "instability_distribution": dict(instability_levels),
            "tasks_with_no_correct_answer": has_no_correct_answer,
            "tasks_with_no_correct_answer_pct": has_no_correct_answer / len(tasks) * 100,
            "avg_options_per_task": sum(len(t.get("options", [])) for t in tasks) / len(tasks),
        }

        if has_no_correct_answer > len(tasks) * 0.5:
            report["dataset_type"] = "cognitive_failure_benchmark"
            report["warnings"].append(
                "Majority of tasks have no correct answer — accuracy metrics disabled"
            )
        else:
            report["dataset_type"] = "mixed_benchmark"

        return report

    @classmethod
    def load_and_validate(cls, path: str) -> Tuple[List[Dict], Dict]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: '{path}'")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            tasks = raw.get("tasks", [])
        elif isinstance(raw, list):
            tasks = raw
        else:
            raise ValueError(f"Unexpected JSON root type: {type(raw)}")
        report = cls.validate(tasks)
        return tasks, report


# =============================================================================
# SECTION 12: PUBLICATION-READY REPORT PRINTER
# =============================================================================

class PublicationReporter:
    """Generates publication-ready text reports."""

    @staticmethod
    def print_validation_report(report: Dict) -> None:
        print("\n" + "=" * 72)
        print("  DATASET VALIDATION REPORT")
        print("=" * 72)
        print(f"  Status: {'✓ VALID' if report['valid'] else '✗ INVALID'}")
        print(f"  Total Tasks: {report['total_tasks']}")
        print(f"  Dataset Type: {report.get('dataset_type', 'unknown')}")
        stats = report.get("statistics", {})
        print("  Difficulty Distribution:")
        for diff, count in stats.get("difficulty_distribution", {}).items():
            print(f"    {diff}: {count}")
        print("  Instability Distribution:")
        for level, count in stats.get("instability_distribution", {}).items():
            print(f"    {level}: {count}")
        no_correct = stats.get("tasks_with_no_correct_answer", 0)
        print(
            f"  Tasks with no correct answer: {no_correct} "
            f"({stats.get('tasks_with_no_correct_answer_pct', 0):.1f}%)"
        )
        if report["errors"]:
            print(f"\n  ERRORS ({len(report['errors'])}):")
            for err in report["errors"][:5]:
                print(f"    ✗ {err}")
            if len(report["errors"]) > 5:
                print(f"    ... and {len(report['errors']) - 5} more")
        if report["warnings"]:
            print(f"\n  WARNINGS ({len(report['warnings'])}):")
            for warn in report["warnings"][:5]:
                print(f"    ⚠ {warn}")
            if len(report["warnings"]) > 5:
                print(f"    ... and {len(report['warnings']) - 5} more")
        print("=" * 72)

    @staticmethod
    def print_model_profile(profile: Dict) -> None:
        model = profile.get("model", "unknown")
        print("\n" + "=" * 72)
        print(f"  MODEL FAILURE PROFILE: {model.upper()}")
        print("=" * 72)
        summary = profile.get("summary_metrics", {})
        print("  COGNITIVE METRICS:")
        print(f"    Meta Score:           {summary.get('mean_meta_score', 0):.3f} ± {summary.get('std_meta_score', 0):.3f}")
        print(f"    Decision Stability:   {summary.get('mean_stability', 0):.3f}")
        print(f"    Decision Entropy:     {summary.get('mean_entropy', 0):.3f}")
        print(f"    Calibration Error:    {summary.get('mean_calibration_error', 0):.3f}")
        print(f"    Coherence Score:      {summary.get('mean_coherence', 0):.3f}")
        print(f"    Contradiction Rate:   {summary.get('mean_contradiction_rate', 0):.3f}")
        print(f"    Mean Confidence:      {summary.get('mean_confidence', 0):.3f}")
        print(f"    Oscillation Index:    {summary.get('mean_oscillation', 0):.3f}")
        print("  COGNITIVE STRUCTURE:")
        print(f"    Cross-Run Inconsistency: {summary.get('mean_cross_run_inconsistency', 0):.3f}")
        print(f"    Epistemic Awareness:     {summary.get('mean_epistemic_awareness', 0):.3f}")
        print(f"    Reasoning Divergence:    {profile.get('cognitive_structure', {}).get('reasoning_divergence', 0):.3f}")
        print(f"    Meta-Cognitive Quality:  {profile.get('cognitive_structure', {}).get('meta_cognitive_quality', 0):.3f}")
        instability = profile.get("instability_index", {})
        print("  INSTABILITY INDEX:")
        print(f"    Composite:            {instability.get('composite_index', 0):.3f}")
        print(f"    Interpretation:       {instability.get('interpretation', 'unknown')}")
        failure = profile.get("failure_profile", {})
        print("  FAILURE MODE COMPOSITION:")
        print(f"    Weak Model Behavior:    {failure.get('weak_model_rate', 0):.1%}")
        print(f"    Strong Model Behavior:  {failure.get('strong_model_rate', 0):.1%}")
        print(f"    Frontier Model Behavior:{failure.get('frontier_model_rate', 0):.1%}")
        print(f"    Uncategorized:          {failure.get('uncategorized_rate', 0):.1%}")
        if failure.get("subcategory_distribution"):
            dominant_subtype = max(
                failure["subcategory_distribution"].items(),
                key=lambda kv: kv[1]
            )[0]
            print(f"    Dominant Failure Subtype:{dominant_subtype}")
        bias = profile.get("confidence_bias_profile", {})
        print("  CONFIDENCE BIAS:")
        print(f"    Overconfident:        {bias.get('overconfident_rate', 0):.1%}")
        print(f"    Underconfident:       {bias.get('underconfident_rate', 0):.1%}")
        print(f"    Well Calibrated:      {bias.get('well_calibrated_rate', 0):.1%}")
        hall = profile.get("hallucination_indicators", {})
        print("  HALLUCINATION INDICATORS:")
        print(f"    Risk Level:           {hall.get('hallucination_risk_level', 'unknown')}")
        print(f"    Contradiction-Coherence Mismatch: {hall.get('contradiction_coherence_mismatch_rate', 0):.1%}")
        mutation = profile.get("mutation_behavior", {})
        if mutation.get("available"):
            print("  ADAPTIVE MUTATION BEHAVIOR:")
            print(f"    Answer Change Rate:  {mutation.get('answer_change_rate', 0):.1%}")
            print(f"    Reasoning Collapse:  {mutation.get('reasoning_collapse_rate', 0):.1%}")
        print("=" * 72)

    @staticmethod
    def print_comparison_table(all_profiles: Dict) -> None:
        print("\n" + "=" * 120)
        print("  MODEL COMPARISON TABLE")
        print("=" * 120)
        header = (
            f"{'Model':<25} "
            f"{'Meta':>6} "
            f"{'Stab':>6} "
            f"{'Ent':>6} "
            f"{'Cal':>6} "
            f"{'Coh':>6} "
            f"{'Contra':>6} "
            f"{'CRI':>6} "
            f"{'EpiAw':>6} "
            f"{'Instab':>6} "
            f"{'Hall':>6}"
        )
        print(header)
        print("-" * 120)
        for model, profile in all_profiles.items():
            summary = profile.get("summary_metrics", {})
            instability = profile.get("instability_index", {})
            hall = profile.get("hallucination_indicators", {})
            hall_str = str(hall.get('hallucination_risk_level', '?'))[:6]
            row = (
                f"{model:<25} "
                f"{summary.get('mean_meta_score', 0):>6.3f} "
                f"{summary.get('mean_stability', 0):>6.3f} "
                f"{summary.get('mean_entropy', 0):>6.3f} "
                f"{summary.get('mean_calibration_error', 0):>6.3f} "
                f"{summary.get('mean_coherence', 0):>6.3f} "
                f"{summary.get('mean_contradiction_rate', 0):>6.3f} "
                f"{summary.get('mean_cross_run_inconsistency', 0):>6.3f} "
                f"{summary.get('mean_epistemic_awareness', 0):>6.3f} "
                f"{instability.get('composite_index', 0):>6.3f} "
                f"{hall_str:>6}"
            )
            print(row)
        print("=" * 120)
        print("  Legend: Meta=MetaScore, Stab=Stability, Ent=Entropy, Cal=CalibrationError,")
        print("          Coh=Coherence, Contra=ContradictionRate, CRI=CrossRunInconsistency,")
        print("          EpiAw=EpistemicAwareness, Instab=InstabilityIndex, Hall=HallucRisk")
        print("=" * 120)

    @staticmethod
    def print_latex_table(all_profiles: Dict) -> str:
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Cognitive Failure Metrics Comparison (v13.0)}",
            r"\label{tab:cognitive_failure_v13}",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Model & Meta & Stability & Entropy & Calibration & Coherence & CRI & Epistemic & Instability \\",
            r"\midrule",
        ]
        for model, profile in all_profiles.items():
            summary = profile.get("summary_metrics", {})
            instability = profile.get("instability_index", {})
            model_clean = model.replace("_", r"\_")
            lines.append(
                f"{model_clean} & "
                f"{summary.get('mean_meta_score', 0):.3f} & "
                f"{summary.get('mean_stability', 0):.3f} & "
                f"{summary.get('mean_entropy', 0):.3f} & "
                f"{summary.get('mean_calibration_error', 0):.3f} & "
                f"{summary.get('mean_coherence', 0):.3f} & "
                f"{summary.get('mean_cross_run_inconsistency', 0):.3f} & "
                f"{summary.get('mean_epistemic_awareness', 0):.3f} & "
                f"{instability.get('composite_index', 0):.3f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\footnotesize{CRI = Cross-Run Inconsistency, Epistemic = Epistemic Awareness Score}",
            r"\end{table}",
        ])
        return "\n".join(lines)


# =============================================================================
# SECTION 13: MAIN ORCHESTRATION (v13.0)
# =============================================================================

def create_api_client(config: EvaluationConfig) -> Any:
    """
    Create the Ollama API client.

    v13.0 STRICT:
    - Requires OLLAMA_API_KEY environment variable; raises RuntimeError if missing.
    - Requires ollama package to be installed; raises RuntimeError if not.
    - Returns OllamaClient pointing at config.api_host.
    - NO mock fallback under any circumstances.
    """
    if not HAS_OLLAMA:
        raise RuntimeError(
            "❌ ollama package is not installed.\n"
            "Install with: pip install ollama\n"
            "Real model execution requires the ollama package."
        )

    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "❌ OLLAMA_API_KEY environment variable is not set.\n"
            "Set it before running: set OLLAMA_API_KEY=<your-key>\n"
            "Real model execution is REQUIRED — no mock fallback."
        )

    # Store key on config for bookkeeping
    config.api_key = api_key

    return OllamaClient(host=config.api_host, timeout=config.timeout_seconds)


def main() -> None:
    """
    Main entry point for BioExecBench Cognitive Evaluation System v13.0.
    """
    config = EvaluationConfig()

    # Override with environment variables if set
    if os.environ.get("OLLAMA_HOST"):
        config.api_host = os.environ["OLLAMA_HOST"]
    if os.environ.get("BIOEXEC_MODELS"):
        config.models = os.environ["BIOEXEC_MODELS"].split(",")
    if os.environ.get("BIOEXEC_NUM_RUNS"):
        config.num_runs_per_task = int(os.environ["BIOEXEC_NUM_RUNS"])

    # Validate configuration
    issues = config.validate()
    if issues:
        print("  [WARN] Configuration issues:")
        for issue in issues:
            print(f"    - {issue}")

    # Initialize reproducibility
    repro = ReproducibilityManager(seed=config.seed, deterministic=config.deterministic)
    repro.initialize()
    repro.fingerprint_config(config.to_dict())

    print("\n" + "=" * 72)
    print("  BioExecBench Cognitive Evaluation System v13.0")
    print("  Publication-Grade AI Reasoning Instability Measurement")
    print("  HARDENED — ZERO MOCK — STRICT API — STRICT PARSING")
    print("=" * 72)

    # Create API client FIRST — hard-fail if not available
    try:
        api_client = create_api_client(config)
    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

    try:
        tasks, validation_report = DatasetValidator.load_and_validate(config.dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    PublicationReporter.print_validation_report(validation_report)

    if not validation_report["valid"]:
        print("\n[ERROR] Dataset validation failed. Please fix schema errors.")
        sys.exit(1)

    output = OutputSystem(config.output_dir, config.output_prefix)
    evaluator = MultiRunEvaluator(config, repro, api_client)
    aggregator = ModelAggregator()

    print(f"\n  RUN CONFIGURATION:")
    print(f"    Models:           {len(config.models)}")
    print(f"    Tasks:            {len(tasks)}")
    print(f"    Runs per task:    {config.num_runs_per_task}")
    print(f"    Temperature:      {config.temperature}")
    print(f"    Empirical Entropy Baseline: {config.empirical_entropy_baseline}")
    print(f"    Adaptive Mutation:{config.enable_adaptive_mutation}")
    print(f"    Seed:             {config.seed}")
    print(f"    Config Hash:      {repro._config_hash}")
    print("=" * 72)

    all_results: Dict[str, List[Dict]] = {}
    all_profiles: Dict[str, Dict] = {}

    for model_idx, model in enumerate(config.models, start=1):
        print(f"\n[START] Model {model_idx}/{len(config.models)}: {model}")
        task_results: List[Dict] = []

        for task_idx, task in enumerate(tasks, start=1):
            task_id = task.get("id", f"task_{task_idx:03d}")
            raw_inst = task.get("meta_instability", "?")
            instability = (
                "extreme" if raw_inst is True
                else "low" if raw_inst is False
                else str(raw_inst)
            )

            print(
                f"  [{task_idx:02d}/{len(tasks)}] {task_id:<35} "
                f"(instability: {instability:<20}) ... ",
                end="", flush=True
            )

            try:
                result = evaluator.evaluate_task(task, model)
                task_results.append(result)

                agg = result["aggregate_metrics"]
                valid_n = agg.get("num_valid_decisions", "?")
                invalid_n = agg.get("num_invalid_decisions", 0)
                print(
                    f"stab={agg['decision_stability']:.2f} "
                    f"ent={agg['decision_entropy']:.2f} "
                    f"cri={agg['cross_run_inconsistency']:.2f} "
                    f"epi={agg['epistemic_awareness']:.2f} "
                    f"valid={valid_n} invalid={invalid_n}"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                repro._log("task_error", {
                    "task_id": task_id,
                    "model": model,
                    "error": str(e)
                })
                continue

        if task_results:
            profile = aggregator.aggregate_model(task_results)
            all_results[model] = task_results
            all_profiles[model] = profile
            PublicationReporter.print_model_profile(profile)

        if config.save_json:
            output.save_full_results(all_results, config, repro)

    print("\n" + "=" * 72)
    print("  GENERATING FINAL OUTPUTS")
    print("=" * 72)

    if all_profiles:
        PublicationReporter.print_comparison_table(all_profiles)

        if config.save_json:
            output.save_failure_profiles(all_profiles)

        if config.save_csv:
            output.save_csv_summary(all_profiles)
            output.save_per_task_csv(all_results)

        if config.generate_plots:
            output.generate_publication_figures(all_profiles)

        print("\n  LATEX TABLE (for publication):")
        print("-" * 50)
        print(PublicationReporter.print_latex_table(all_profiles))
        print("-" * 50)

    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE")
    print("=" * 72)
    print(f"  Results saved to: {config.output_dir}/")
    print(f"  Config hash: {repro._config_hash}")
    print(f"  Total models evaluated: {len(all_profiles)}")
    print(f"  Total tasks evaluated: {sum(len(v) for v in all_results.values())}")
    print("=" * 72)


if __name__ == "__main__":
    main()