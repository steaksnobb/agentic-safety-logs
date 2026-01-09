# Agentic-Safety-Audit

## Abstract: Logic Drift in Monte Carlo Simulations with Wyom Gamma

This project implements a comprehensive safety auditing framework designed to stress-test AI code generation systems, specifically Claude Code, for logic drift vulnerabilities. Logic drift occurs when AI-generated code refactorings inadvertently introduce subtle mathematical or logical errors that deviate from the original implementation's behavior.

### Wyom Gamma Monte Carlo Framework

The Wyom Gamma methodology employs Monte Carlo simulations to systematically probe for logic drift in financial mathematics implementations. Our approach focuses on Black-Scholes option pricing models as a canonical test case, where even microscopic numerical deviations can compound into significant pricing errors.

**Key Concepts:**

1. **Logic Drift Detection**: Automated comparison of original mathematical implementations against AI-refactored versions using statistical divergence metrics.

2. **Monte Carlo Validation**: Thousands of randomized parameter combinations are tested to identify edge cases where refactored code produces divergent outputs.

3. **Wyom Gamma Risk Profile**: A specialized risk quantification metric that measures the sensitivity of pricing functions to parameter variations, derived from the traditional options Greek "Gamma" but extended to capture higher-order logic stability.

4. **Immutable Safety Constitution**: Predefined constraints and forbidden patterns that any code modification must respect, encoded in `config/risk_constitution.yaml`.

### Architecture

```
agentic-safety-logs/
├── config/
│   └── risk_constitution.yaml    # Immutable safety constraints
├── harness/
│   └── drift_detector.py          # LogicDriftAnalyzer core logic
├── telemetry/
│   └── hallucination_log.json    # Anomaly detection logs
└── requirements.txt               # Python dependencies
```

### Usage

```python
from harness.drift_detector import LogicDriftAnalyzer

# Initialize analyzer
analyzer = LogicDriftAnalyzer(
    original_implementation=black_scholes_original,
    refactored_implementation=black_scholes_refactored
)

# Run Monte Carlo drift analysis
results = analyzer.run_drift_analysis(n_simulations=10000)

# Check for violations
if results.has_violations():
    print(f"Logic drift detected: {results.get_summary()}")
```

### Installation

```bash
pip install -r requirements.txt
```

### Research Context

Logic drift represents a critical safety concern in AI-assisted code generation. While traditional testing validates functional correctness for known inputs, Monte Carlo approaches reveal statistical distributions of error across the input space. The Wyom Gamma framework specifically targets scenarios where:

- Numerical precision differences accumulate
- Mathematical identities are violated through algebraic manipulation
- Boundary conditions are mishandled in edge cases
- Optimization shortcuts introduce approximation errors

This research toolkit enables systematic evaluation of AI code generation safety, providing quantitative metrics for logic stability under automated refactoring.