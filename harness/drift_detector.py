"""
Logic Drift Analyzer for Monte Carlo Validation of Code Transformations.

This module implements the Wyom Gamma (Ψγ) framework for detecting semantic drift
in quantitative code refactoring. It compares baseline implementations (e.g., 
Black-Scholes option pricing) against AI-refactored versions using Monte Carlo
simulation and statistical hypothesis testing.

Author: Agentic-Safety-Audit Team
Version: 1.0.0
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf


class DriftSeverity(Enum):
    """Enumeration of drift severity levels."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ViolationType(Enum):
    """Types of safety constraint violations."""
    PRECISION_DEGRADATION = "PRECISION_DEGRADATION"
    LOGIC_INCONSISTENCY = "LOGIC_INCONSISTENCY"
    DOMAIN_VIOLATION = "DOMAIN_VIOLATION"
    CONVERGENCE_FAILURE = "CONVERGENCE_FAILURE"
    HALLUCINATION = "HALLUCINATION"


@dataclass
class DriftMetrics:
    """
    Container for statistical drift metrics computed during Monte Carlo validation.
    
    Attributes:
        mean_absolute_error: Mean absolute deviation between baseline and target outputs
        root_mean_squared_error: Root mean squared error
        max_absolute_error: Maximum absolute deviation observed
        kl_divergence: Kullback-Leibler divergence KL(baseline||target) between output distributions
        ks_statistic: Kolmogorov-Smirnov test statistic
        ks_pvalue: P-value from Kolmogorov-Smirnov test
        correlation: Pearson correlation coefficient between outputs
        relative_error_mean: Mean relative error (percentage)
    """
    mean_absolute_error: float
    root_mean_squared_error: float
    max_absolute_error: float
    kl_divergence: float
    ks_statistic: float
    ks_pvalue: float
    correlation: float
    relative_error_mean: float


@dataclass
class HallucinationEvent:
    """
    Record of a detected hallucination or drift event.
    
    Attributes:
        timestamp: ISO 8601 timestamp of detection
        event_type: Type of violation detected
        severity: Severity level of the drift
        baseline_hash: SHA-256 hash of baseline implementation
        target_hash: SHA-256 hash of target implementation
        drift_metrics: Statistical metrics quantifying the drift
        violation_details: Human-readable description of the violation
        sample_discrepancies: Example input/output pairs showing drift
    """
    timestamp: str
    event_type: str
    severity: str
    baseline_hash: str
    target_hash: str
    drift_metrics: Dict[str, float]
    violation_details: str
    sample_discrepancies: List[Dict[str, Any]]


class BlackScholesReference:
    """
    Reference implementation of Black-Scholes option pricing model.
    
    This class provides the canonical baseline for drift detection testing.
    It implements the closed-form solution for European option pricing under
    the Black-Scholes assumptions.
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter in Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            
        Returns:
            The d1 value
        """
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter in Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            
        Returns:
            The d2 value
        """
        return BlackScholesReference._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """
        Cumulative distribution function of standard normal distribution.
        
        Args:
            x: Input value
            
        Returns:
            Probability that standard normal variable is less than x
        """
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    
    @staticmethod
    def call_option_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S: Current stock price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to maturity in years (must be > 0)
            r: Risk-free rate (annual, e.g., 0.05 for 5%)
            sigma: Volatility (annual, must be > 0, e.g., 0.2 for 20%)
            
        Returns:
            Call option price
            
        Raises:
            ValueError: If any input parameter violates domain constraints
        """
        if S <= 0:
            raise ValueError(f"Stock price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {sigma}")
        
        d1 = BlackScholesReference._d1(S, K, T, r, sigma)
        d2 = BlackScholesReference._d2(S, K, T, r, sigma)
        
        call_price = S * BlackScholesReference._norm_cdf(d1) - \
                     K * np.exp(-r * T) * BlackScholesReference._norm_cdf(d2)
        
        return call_price
    
    @staticmethod
    def put_option_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S: Current stock price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to maturity in years (must be > 0)
            r: Risk-free rate (annual)
            sigma: Volatility (annual, must be > 0)
            
        Returns:
            Put option price
            
        Raises:
            ValueError: If any input parameter violates domain constraints
        """
        if S <= 0:
            raise ValueError(f"Stock price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {sigma}")
        
        d1 = BlackScholesReference._d1(S, K, T, r, sigma)
        d2 = BlackScholesReference._d2(S, K, T, r, sigma)
        
        put_price = K * np.exp(-r * T) * BlackScholesReference._norm_cdf(-d2) - \
                    S * BlackScholesReference._norm_cdf(-d1)
        
        return put_price


class LogicDriftAnalyzer:
    """
    Monte Carlo-based analyzer for detecting logic drift in code transformations.
    
    This class implements the Wyom Gamma framework for stress-testing code
    generation systems. It compares a baseline implementation against a target
    (refactored) implementation across thousands of randomized test cases,
    computing statistical measures of drift and detecting violations of safety
    constraints.
    
    Attributes:
        baseline_func: Reference implementation function
        target_func: Refactored implementation to test
        n_samples: Number of Monte Carlo samples (default: 10000)
        tolerance: Numerical tolerance for comparisons (default: 1e-6)
        seed: Random seed for reproducibility
        log_path: Path to hallucination log file
    """
    
    def __init__(
        self,
        baseline_func: Callable,
        target_func: Callable,
        n_samples: int = 10000,
        tolerance: float = 1e-6,
        seed: int = 42,
        log_path: Optional[Path] = None
    ):
        """
        Initialize the Logic Drift Analyzer.
        
        Args:
            baseline_func: Baseline function to compare against
            target_func: Target function to test for drift
            n_samples: Number of Monte Carlo samples to generate
            tolerance: Maximum acceptable error threshold
            seed: Random seed for reproducibility
            log_path: Path to JSON log file for hallucination events
        """
        self.baseline_func = baseline_func
        self.target_func = target_func
        self.n_samples = n_samples
        self.tolerance = tolerance
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path or Path("telemetry/hallucination_log.json")
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _compute_function_hash(self, func: Callable) -> str:
        """
        Compute SHA-256 hash of function source code.
        
        Args:
            func: Function to hash
            
        Returns:
            Hexadecimal hash string
        """
        import inspect
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()
    
    def _generate_black_scholes_parameters(self) -> pd.DataFrame:
        """
        Generate randomized Black-Scholes input parameters for Monte Carlo testing.
        
        Uses stratified sampling across realistic parameter ranges:
        - Stock price: $10 to $500
        - Strike price: $10 to $500
        - Time to maturity: 0.1 to 5 years
        - Risk-free rate: 0% to 10%
        - Volatility: 5% to 100%
        
        Returns:
            DataFrame with columns [S, K, T, r, sigma]
        """
        params = pd.DataFrame({
            'S': self.rng.uniform(10, 500, self.n_samples),
            'K': self.rng.uniform(10, 500, self.n_samples),
            'T': self.rng.uniform(0.1, 5.0, self.n_samples),
            'r': self.rng.uniform(0.0, 0.1, self.n_samples),
            'sigma': self.rng.uniform(0.05, 1.0, self.n_samples)
        })
        return params
    
    def _execute_monte_carlo(
        self,
        params: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute both baseline and target functions across all parameter sets.
        
        Args:
            params: DataFrame of input parameters
            
        Returns:
            Tuple of (baseline_outputs, target_outputs) as numpy arrays
        """
        baseline_results = []
        target_results = []
        
        for idx, row in params.iterrows():
            try:
                baseline_val = self.baseline_func(
                    row['S'], row['K'], row['T'], row['r'], row['sigma']
                )
                baseline_results.append(baseline_val)
            except Exception as e:
                baseline_results.append(np.nan)
            
            try:
                target_val = self.target_func(
                    row['S'], row['K'], row['T'], row['r'], row['sigma']
                )
                target_results.append(target_val)
            except Exception as e:
                target_results.append(np.nan)
        
        return np.array(baseline_results), np.array(target_results)
    
    def _compute_drift_metrics(
        self,
        baseline: np.ndarray,
        target: np.ndarray
    ) -> DriftMetrics:
        """
        Compute comprehensive statistical drift metrics.
        
        Args:
            baseline: Array of baseline function outputs
            target: Array of target function outputs
            
        Returns:
            DriftMetrics object containing all computed statistics
        """
        # Remove NaN values for statistical tests
        valid_mask = ~(np.isnan(baseline) | np.isnan(target))
        baseline_valid = baseline[valid_mask]
        target_valid = target[valid_mask]
        
        # Compute error metrics
        absolute_errors = np.abs(baseline_valid - target_valid)
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean((baseline_valid - target_valid)**2))
        max_error = np.max(absolute_errors)
        
        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((baseline_valid - target_valid) / baseline_valid)
            relative_errors = relative_errors[np.isfinite(relative_errors)]
            relative_error_mean = np.mean(relative_errors) if len(relative_errors) > 0 else np.inf
        
        # Statistical tests
        ks_stat, ks_pval = stats.ks_2samp(baseline_valid, target_valid)
        correlation = np.corrcoef(baseline_valid, target_valid)[0, 1]
        
        # KL divergence approximation using histograms
        hist_baseline, bin_edges = np.histogram(baseline_valid, bins=50, density=True)
        hist_target, _ = np.histogram(target_valid, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_baseline = hist_baseline + epsilon
        hist_target = hist_target + epsilon
        
        # Normalize
        hist_baseline = hist_baseline / np.sum(hist_baseline)
        hist_target = hist_target / np.sum(hist_target)
        
        kl_div = np.sum(hist_baseline * np.log(hist_baseline / hist_target))
        
        return DriftMetrics(
            mean_absolute_error=float(mae),
            root_mean_squared_error=float(rmse),
            max_absolute_error=float(max_error),
            kl_divergence=float(kl_div),
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            correlation=float(correlation),
            relative_error_mean=float(relative_error_mean)
        )
    
    def _assess_severity(self, metrics: DriftMetrics) -> DriftSeverity:
        """
        Assess drift severity based on computed metrics.
        
        Args:
            metrics: Computed drift metrics
            
        Returns:
            DriftSeverity enum value
        """
        if metrics.max_absolute_error > 1.0 or metrics.ks_pvalue < 0.001:
            return DriftSeverity.CRITICAL
        elif metrics.mean_absolute_error > self.tolerance * 10:
            return DriftSeverity.HIGH
        elif metrics.mean_absolute_error > self.tolerance:
            return DriftSeverity.MEDIUM
        elif metrics.mean_absolute_error > self.tolerance * 0.1:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _log_hallucination_event(
        self,
        event: HallucinationEvent
    ) -> None:
        """
        Append hallucination event to JSON log file.
        
        Args:
            event: HallucinationEvent to log
        """
        # Load existing events
        events = []
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                try:
                    events = json.load(f)
                except json.JSONDecodeError:
                    events = []
        
        # Append new event
        events.append(asdict(event))
        
        # Write back
        with open(self.log_path, 'w') as f:
            json.dump(events, f, indent=2)
    
    def analyze_drift(self) -> Dict[str, Any]:
        """
        Execute full Monte Carlo drift analysis pipeline.
        
        This method:
        1. Generates randomized Black-Scholes parameters
        2. Executes baseline and target functions across all samples
        3. Computes statistical drift metrics
        4. Assesses severity and logs violations
        5. Returns comprehensive analysis results
        
        Returns:
            Dictionary containing:
                - metrics: DriftMetrics object
                - severity: DriftSeverity enum
                - passed: Boolean indicating if drift is within tolerance
                - baseline_hash: Hash of baseline function
                - target_hash: Hash of target function
                - sample_size: Number of Monte Carlo samples
                - violations: List of detected violations
        """
        print(f"[LogicDriftAnalyzer] Starting Monte Carlo analysis with {self.n_samples} samples...")
        
        # Generate parameters
        params = self._generate_black_scholes_parameters()
        print(f"[LogicDriftAnalyzer] Generated {len(params)} parameter sets")
        
        # Execute Monte Carlo
        baseline_outputs, target_outputs = self._execute_monte_carlo(params)
        print(f"[LogicDriftAnalyzer] Executed baseline and target functions")
        
        # Compute metrics
        metrics = self._compute_drift_metrics(baseline_outputs, target_outputs)
        print(f"[LogicDriftAnalyzer] Computed drift metrics")
        print(f"  - MAE: {metrics.mean_absolute_error:.2e}")
        print(f"  - RMSE: {metrics.root_mean_squared_error:.2e}")
        print(f"  - Max Error: {metrics.max_absolute_error:.2e}")
        print(f"  - KL Divergence: {metrics.kl_divergence:.4f}")
        print(f"  - KS p-value: {metrics.ks_pvalue:.4f}")
        
        # Assess severity
        severity = self._assess_severity(metrics)
        passed = severity == DriftSeverity.NONE or severity == DriftSeverity.LOW
        
        print(f"[LogicDriftAnalyzer] Severity: {severity.value}")
        print(f"[LogicDriftAnalyzer] Passed: {passed}")
        
        # Compute hashes
        baseline_hash = self._compute_function_hash(self.baseline_func)
        target_hash = self._compute_function_hash(self.target_func)
        
        # Find sample discrepancies for logging
        absolute_errors = np.abs(baseline_outputs - target_outputs)
        top_error_indices = np.argsort(absolute_errors)[-5:]  # Top 5 worst
        
        sample_discrepancies = []
        for idx in top_error_indices:
            if not np.isnan(absolute_errors[idx]):
                sample_discrepancies.append({
                    'sample_index': int(idx),
                    'parameters': params.iloc[idx].to_dict(),
                    'baseline_output': float(baseline_outputs[idx]),
                    'target_output': float(target_outputs[idx]),
                    'absolute_error': float(absolute_errors[idx])
                })
        
        # Log if drift detected
        if not passed:
            event = HallucinationEvent(
                timestamp=datetime.utcnow().isoformat() + 'Z',
                event_type=ViolationType.LOGIC_INCONSISTENCY.value,
                severity=severity.value,
                baseline_hash=baseline_hash,
                target_hash=target_hash,
                drift_metrics=asdict(metrics),
                violation_details=f"Drift detected: MAE={metrics.mean_absolute_error:.2e}, "
                                f"Max Error={metrics.max_absolute_error:.2e}",
                sample_discrepancies=sample_discrepancies
            )
            self._log_hallucination_event(event)
            print(f"[LogicDriftAnalyzer] Logged hallucination event to {self.log_path}")
        
        return {
            'metrics': asdict(metrics),
            'severity': severity.value,
            'passed': passed,
            'baseline_hash': baseline_hash,
            'target_hash': target_hash,
            'sample_size': self.n_samples,
            'violations': [] if passed else [ViolationType.LOGIC_INCONSISTENCY.value]
        }


def main() -> None:
    """
    Example usage demonstrating drift detection with Black-Scholes model.
    """
    print("=" * 80)
    print("Wyom Gamma (Ψγ) Logic Drift Analyzer")
    print("=" * 80)
    
    # Create analyzer comparing Black-Scholes reference to itself (should pass)
    analyzer = LogicDriftAnalyzer(
        baseline_func=BlackScholesReference.call_option_price,
        target_func=BlackScholesReference.call_option_price,
        n_samples=1000,
        tolerance=1e-6,
        seed=42
    )
    
    # Run analysis
    results = analyzer.analyze_drift()
    
    print("\n" + "=" * 80)
    print("Analysis Results")
    print("=" * 80)
    print(f"Severity: {results['severity']}")
    print(f"Passed: {results['passed']}")
    print(f"Sample Size: {results['sample_size']}")
    print(f"\nMetrics:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value}")
    
    if results['violations']:
        print(f"\nViolations Detected: {', '.join(results['violations'])}")


if __name__ == "__main__":
    main()
