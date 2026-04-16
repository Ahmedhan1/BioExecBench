# BioExecBench
### Project Name

**BioExecBench: A Failure-Oriented Cognitive Benchmark for Measuring Instability in Frontier Models**

---

### Your Team

Dr. Ahmed Hani (Independent Researcher)

---

### Problem Statement

Current LLM benchmarks primarily measure **accuracy**, but fail to evaluate how models behave when confronted with **unsolvable, contradictory, or adversarial scenarios**. This creates a critical scientific blind spot:

> Models can appear highly capable while systematically failing in underlying reasoning.

This project introduces a **failure-oriented evaluation paradigm** designed to measure:

* Cognitive instability
* Epistemic awareness
* Reasoning consistency

These dimensions align directly with key capabilities outlined in the AGI evaluation framework.

Unlike traditional benchmarks, BioExecBench **removes the notion of a correct answer**, isolating how models reason rather than what they know.

---

### Task & Benchmark Construction

BioExecBench consists of **20 adversarial biomedical tasks**, each engineered to induce a specific cognitive failure mode.

#### Core Design Innovation

Each task enforces **ONE dominant failure axis** (e.g., observer effect, rule conflict, temporal instability).

Tasks are intentionally **unsolvable by design**, incorporating:

* Multi-layer contradictions
* Hidden invalid assumptions

This results in:

* Clear failure attribution
* Reduced evaluation noise
* Strong interpretability

---

#### Evaluation Protocol

* **Models Evaluated:**
  qwen3.5:397b-cloud, gemma4:31b-cloud, kimi-k2.5:cloud

* **Execution:**
  7 runs per task (cross-run statistical aggregation)

* **Configuration:**
  Temperature = 0.2 (controlled reasoning)
  Adaptive mutation = enabled

---

### Dataset

* **Total Tasks:** 20
* **Domain:** Biomedical & clinical reasoning
* **Difficulty:** Extreme (100% unsolvable)

Example domains include:

* CRISPR off-target analysis
* Phase III trial futility
* Federated learning failure scenarios

Each task contains:

* Adversarial scenario
* 5-option decision space (A–E)

---

#### Key Property

> Any confident answer is inherently suspect.

This transforms the benchmark into a probe for:

* Overconfidence
* Failure detection
* Meta-reasoning capability

---

### Technical Details

#### 1. Robust Decision Extraction Engine

Adversarial prompting produced up to **100% format non-compliance** in raw outputs.

To address this, we implemented a **multi-layer parsing system**:

* JSON extraction
* Pattern-based detection (Decision / Final Answer)
* Heuristic token-level fallback

→ Reduced invalid outputs from **100% → near-zero usable signal loss**

---

#### 2. Cognitive Metrics

The system computes a **multi-dimensional cognitive profile**:

* Decision Stability
* Decision Entropy
* Cross-Run Inconsistency (CRI)
* Calibration Error
* Epistemic Awareness

This shifts evaluation from **accuracy → reasoning behavior**.

---

### Results, Insights, and Conclusions

The benchmark was executed across **60 total evaluations (20 tasks × 3 models)**.

#### Model Comparison

| Model        | Meta Score | Stability | Entropy | Calibration | CRI   | Epistemic Awareness |
| ------------ | ---------- | --------- | ------- | ----------- | ----- | ------------------- |
| qwen3.5:397b | 0.540      | 0.988     | 0.041   | 0.376       | 0.267 | 0.000               |
| gemma4:31b   | 0.552      | 0.979     | 0.073   | 0.294       | 0.518 | 0.000               |
| kimi-k2.5    | 0.551      | 0.954     | 0.145   | 0.277       | 0.558 | 0.000               |

---

### Key Findings

#### 1. Zero Epistemic Awareness

Across all models and tasks:

> Epistemic Awareness = **0.000**

Models never detect that the underlying problem is invalid.

---

#### 2. Consistent Failure

* Stability: **0.954 – 0.988**
* Entropy: **near zero**

Models do not fail randomly —
they fail **consistently and deterministically**.

---

#### 3. Structural Overconfidence

Despite contradictory inputs:

* Models produce stable decisions
* Fail to question assumptions

→ Indicates absence of internal validation mechanisms

---

#### 4. Benchmark Discriminatory Power

BioExecBench distinguishes between:

* Random hallucination
* Structured cognitive collapse

This capability is not captured by standard benchmarks.

---

### Novel Contribution

This work introduces a new evaluation paradigm:

> **Failure is treated as signal, not noise**

BioExecBench reframes evaluation from:

* correctness → reasoning behavior
* output → cognitive process

---

### Organizational Affiliations

Independent Research / SciPharma

---

### References & Citations

* Google DeepMind — *Measuring Progress Toward AGI: A Cognitive Framework*
* Kaggle Benchmarks SDK

---

### Final Statement

BioExecBench demonstrates that:

> Frontier models are not limited by knowledge—
> but by how they reason under uncertainty.

---

**Core Insight:**

> The future of evaluation lies not in measuring correctness,
> but in understanding **how and why models fail**.

