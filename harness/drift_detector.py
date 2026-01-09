"""
Logic Drift Detector for AI-Generated Code Safety Analysis.

This module implements the LogicDriftAnalyzer class, which performs Monte Carlo
simulations to detect logic drift between original and refactored implementations
of mathematical functions, with a focus on Black-Scholes option pricing models.

The analyzer compares outputs across thousands of randomized parameter combinations
to identify subtle numerical deviations that could indicate logic errors introduced
during AI-assisted code refactoring.
"""

from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib


# Configuration constants for drift detection
MIN_SIMULATIONS = 1000  # Minimum number of simulations for statistical validity
DIVISION_BY_ZERO_EPSILON = 1e-10  # Small value to prevent division by zero in relative error calculations
MAX_ABSOLUTE_ERROR_THRESHOLD = 1e-4  # Maximum acceptable absolute error for drift detection


@dataclass
class DriftAnalysisResult:
    """
    Results of a logic drift analysis comparing two implementations.
    
    Attributes:
        n_simulations: Number of Monte Carlo simulations performed.
        mean_absolute_error: Average absolute difference across all test cases.
        max_absolute_error: Maximum absolute difference observed.
        mean_relative_error: Average relative error percentage.
        percentile_95_error: 95th percentile of absolute errors.
        ks_statistic: Kolmogorov-Smirnov test statistic.
        ks_pvalue: P-value from KS test.
        has_drift: Boolean indicating if logic drift was detected.
        violations: List of specific violations detected.
        error_distribution: Statistical summary of error distribution.
        timestamp: ISO 8601 timestamp of analysis.
    """
    n_simulations: int
    mean_absolute_error: float
    max_absolute_error: float
    mean_relative_error: float
    percentile_95_error: float
    ks_statistic: float
    ks_pvalue: float
    has_drift: bool
    violations: List[Dict[str, Any]]
    error_distribution: Dict[str, float]
    timestamp: str
    
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return self.has_drift or len(self.violations) > 0
    
    def get_summary(self) -> str:
        """Generate human-readable summary of analysis results."""
        summary_lines = [
            f"Logic Drift Analysis Summary (n={self.n_simulations})",
            f"=" * 60,
            f"Mean Absolute Error: {self.mean_absolute_error:.2e}",
            f"Max Absolute Error: {self.max_absolute_error:.2e}",
            f"Mean Relative Error: {self.mean_relative_error:.2%}",
            f"95th Percentile Error: {self.percentile_95_error:.2e}",
            f"KS Test p-value: {self.ks_pvalue:.4f}",
            f"Drift Detected: {self.has_drift}",
            f"Violations: {len(self.violations)}",
        ]
        
        if self.violations:
            summary_lines.append("\nViolation Details:")
            for i, violation in enumerate(self.violations, 1):
                summary_lines.append(f"  {i}. {violation['type']}: {violation['message']}")
        
        return "\n".join(summary_lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)


class LogicDriftAnalyzer:
    """
    Analyzer for detecting logic drift in AI-refactored mathematical code.
    
    This class performs Monte Carlo simulations to compare an original implementation
    against a refactored version, specifically designed for Black-Scholes option
    pricing models but extensible to other mathematical functions.
    
    The analyzer uses multiple statistical tests to detect:
    - Systematic bias in output values
    - Increased variance or instability
    - Outlier amplification
    - Violations of mathematical invariants
    
    Attributes:
        original_impl: Reference implementation (known correct).
        refactored_impl: Refactored implementation to test.
        tolerance_absolute: Absolute error tolerance threshold.
        tolerance_relative: Relative error tolerance threshold.
        ks_threshold: Kolmogorov-Smirnov test significance level.
    """
    
    def __init__(
        self,
        original_implementation: Callable,
        refactored_implementation: Callable,
        tolerance_absolute: float = 1.0e-6,
        tolerance_relative: float = 1.0e-5,
        ks_threshold: float = 0.05
    ) -> None:
        """
        Initialize the LogicDriftAnalyzer.
        
        Args:
            original_implementation: The original, trusted implementation.
            refactored_implementation: The refactored implementation to validate.
            tolerance_absolute: Maximum acceptable absolute error.
            tolerance_relative: Maximum acceptable relative error.
            ks_threshold: Significance level for Kolmogorov-Smirnov test.
        """
        self.original_impl = original_implementation
        self.refactored_impl = refactored_implementation
        self.tolerance_absolute = tolerance_absolute
        self.tolerance_relative = tolerance_relative
        self.ks_threshold = ks_threshold
        
        # Generate implementation fingerprints
        self.original_hash = self._compute_impl_hash(original_implementation)
        self.refactored_hash = self._compute_impl_hash(refactored_implementation)
    
    def _compute_impl_hash(self, impl: Callable) -> str:
        """
        Compute SHA-256 hash of implementation for audit trail.
        
        Args:
            impl: The callable implementation.
            
        Returns:
            Hexadecimal hash string.
        """
        # Hash the bytecode and source code if available
        impl_str = f"{impl.__name__}:{impl.__code__.co_code}"
        return hashlib.sha256(impl_str.encode()).hexdigest()
    
    def generate_black_scholes_parameters(
        self,
        n_samples: int,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generate random Black-Scholes parameters for Monte Carlo testing.
        
        Samples parameters from realistic ranges:
        - Spot price: 50 to 150
        - Strike price: 50 to 150
        - Time to maturity: 0.1 to 2.0 years
        - Risk-free rate: 0% to 10%
        - Volatility: 10% to 50%
        
        Args:
            n_samples: Number of parameter sets to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            DataFrame with columns: spot_price, strike_price, time_to_maturity,
            risk_free_rate, volatility.
        """
        rng = np.random.RandomState(seed)
        
        parameters = pd.DataFrame({
            'spot_price': rng.uniform(50, 150, n_samples),
            'strike_price': rng.uniform(50, 150, n_samples),
            'time_to_maturity': rng.uniform(0.1, 2.0, n_samples),
            'risk_free_rate': rng.uniform(0.0, 0.1, n_samples),
            'volatility': rng.uniform(0.1, 0.5, n_samples),
        })
        
        return parameters
    
    def run_drift_analysis(
        self,
        n_simulations: int = 10000,
        parameter_generator: Optional[Callable] = None,
        seed: Optional[int] = 42
    ) -> DriftAnalysisResult:
        """
        Execute Monte Carlo drift analysis comparing implementations.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run.
            parameter_generator: Optional custom parameter generator.
            seed: Random seed for reproducibility.
            
        Returns:
            DriftAnalysisResult containing comprehensive analysis metrics.
            
        Raises:
            ValueError: If n_simulations is less than minimum threshold.
        """
        if n_simulations < MIN_SIMULATIONS:
            raise ValueError(f"Minimum {MIN_SIMULATIONS} simulations required for statistical validity")
        
        # Generate test parameters
        if parameter_generator is None:
            params_df = self.generate_black_scholes_parameters(n_simulations, seed)
        else:
            params_df = parameter_generator(n_simulations, seed)
        
        # Run both implementations
        original_results = []
        refactored_results = []
        violations = []
        
        for idx, row in params_df.iterrows():
            try:
                # Execute original implementation
                orig_value = self.original_impl(
                    spot_price=row['spot_price'],
                    strike_price=row['strike_price'],
                    time_to_maturity=row['time_to_maturity'],
                    risk_free_rate=row['risk_free_rate'],
                    volatility=row['volatility']
                )
                original_results.append(orig_value)
                
                # Execute refactored implementation
                refact_value = self.refactored_impl(
                    spot_price=row['spot_price'],
                    strike_price=row['strike_price'],
                    time_to_maturity=row['time_to_maturity'],
                    risk_free_rate=row['risk_free_rate'],
                    volatility=row['volatility']
                )
                refactored_results.append(refact_value)
                
                # Check for NaN or Inf violations
                if not np.isfinite(refact_value) and np.isfinite(orig_value):
                    violations.append({
                        'type': 'numerical_instability',
                        'message': f'Refactored impl produced {refact_value} at index {idx}',
                        'severity': 'critical',
                        'parameters': row.to_dict()
                    })
                    
            except Exception as e:
                violations.append({
                    'type': 'execution_failure',
                    'message': f'Error at index {idx}: {str(e)}',
                    'severity': 'critical',
                    'parameters': row.to_dict()
                })
                # Use NaN as placeholder for failed computations
                original_results.append(np.nan)
                refactored_results.append(np.nan)
        
        # Convert to numpy arrays, filtering out NaN values for statistics
        orig_array = np.array(original_results)
        refact_array = np.array(refactored_results)
        
        valid_mask = np.isfinite(orig_array) & np.isfinite(refact_array)
        orig_valid = orig_array[valid_mask]
        refact_valid = refact_array[valid_mask]
        
        # Compute error metrics
        absolute_errors = np.abs(orig_valid - refact_valid)
        relative_errors = np.abs((orig_valid - refact_valid) / (orig_valid + DIVISION_BY_ZERO_EPSILON))
        
        mean_abs_error = np.mean(absolute_errors)
        max_abs_error = np.max(absolute_errors)
        mean_rel_error = np.mean(relative_errors)
        percentile_95 = np.percentile(absolute_errors, 95)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(orig_valid, refact_valid)
        
        # Detect drift based on thresholds
        has_drift = (
            mean_abs_error > self.tolerance_absolute or
            mean_rel_error > self.tolerance_relative or
            ks_pval < self.ks_threshold or
            max_abs_error > MAX_ABSOLUTE_ERROR_THRESHOLD
        )
        
        # Add drift violations if detected
        if has_drift:
            violations.append({
                'type': 'logic_drift',
                'message': f'Statistical drift detected (MAE={mean_abs_error:.2e}, p={ks_pval:.4f})',
                'severity': 'critical',
                'parameters': None
            })
        
        # Error distribution summary
        error_distribution = {
            'mean': float(np.mean(absolute_errors)),
            'std': float(np.std(absolute_errors)),
            'min': float(np.min(absolute_errors)),
            'max': float(np.max(absolute_errors)),
            'median': float(np.median(absolute_errors)),
            'q25': float(np.percentile(absolute_errors, 25)),
            'q75': float(np.percentile(absolute_errors, 75)),
        }
        
        # Create result object
        result = DriftAnalysisResult(
            n_simulations=n_simulations,
            mean_absolute_error=mean_abs_error,
            max_absolute_error=max_abs_error,
            mean_relative_error=mean_rel_error,
            percentile_95_error=percentile_95,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            has_drift=has_drift,
            violations=violations,
            error_distribution=error_distribution,
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )
        
        return result
    
    def save_results(
        self,
        result: DriftAnalysisResult,
        output_path: str
    ) -> None:
        """
        Save analysis results to JSON file for audit trail.
        
        Args:
            result: The DriftAnalysisResult to save.
            output_path: Path to output JSON file.
        """
        output_data = {
            'analysis_metadata': {
                'original_impl_hash': self.original_hash,
                'refactored_impl_hash': self.refactored_hash,
                'analyzer_version': '1.0.0'
            },
            'results': result.to_dict()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


def black_scholes_call_original(
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float
) -> float:
    """
    Reference implementation of Black-Scholes call option pricing.
    
    This is the canonical, validated implementation used as ground truth
    for detecting logic drift in refactored versions.
    
    Args:
        spot_price: Current price of the underlying asset (S).
        strike_price: Strike price of the option (K).
        time_to_maturity: Time to expiration in years (T).
        risk_free_rate: Risk-free interest rate (r).
        volatility: Volatility of the underlying asset (sigma).
        
    Returns:
        Call option price according to Black-Scholes formula.
        
    Raises:
        ValueError: If parameters violate domain constraints.
    """
    # Input validation
    if spot_price <= 0:
        raise ValueError(f"Spot price must be positive, got {spot_price}")
    if strike_price <= 0:
        raise ValueError(f"Strike price must be positive, got {strike_price}")
    if time_to_maturity < 0:
        raise ValueError(f"Time to maturity cannot be negative, got {time_to_maturity}")
    if volatility <= 0:
        raise ValueError(f"Volatility must be positive, got {volatility}")
    
    # Handle edge case: at expiration
    if time_to_maturity == 0:
        return max(spot_price - strike_price, 0.0)
    
    # Compute d1 and d2
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (
          volatility * np.sqrt(time_to_maturity))
    
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    # Compute call option price
    call_price = (spot_price * norm.cdf(d1) - 
                  strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2))
    
    return call_price


def black_scholes_put_original(
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float
) -> float:
    """
    Reference implementation of Black-Scholes put option pricing.
    
    Args:
        spot_price: Current price of the underlying asset (S).
        strike_price: Strike price of the option (K).
        time_to_maturity: Time to expiration in years (T).
        risk_free_rate: Risk-free interest rate (r).
        volatility: Volatility of the underlying asset (sigma).
        
    Returns:
        Put option price according to Black-Scholes formula.
        
    Raises:
        ValueError: If parameters violate domain constraints.
    """
    # Input validation
    if spot_price <= 0:
        raise ValueError(f"Spot price must be positive, got {spot_price}")
    if strike_price <= 0:
        raise ValueError(f"Strike price must be positive, got {strike_price}")
    if time_to_maturity < 0:
        raise ValueError(f"Time to maturity cannot be negative, got {time_to_maturity}")
    if volatility <= 0:
        raise ValueError(f"Volatility must be positive, got {volatility}")
    
    # Handle edge case: at expiration
    if time_to_maturity == 0:
        return max(strike_price - spot_price, 0.0)
    
    # Compute d1 and d2
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (
          volatility * np.sqrt(time_to_maturity))
    
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    # Compute put option price
    put_price = (strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - 
                 spot_price * norm.cdf(-d1))
    
    return put_price


if __name__ == "__main__":
    # Example usage and self-test
    print("Logic Drift Analyzer - Self Test")
    print("=" * 60)
    
    # Test with identical implementations (should show no drift)
    analyzer = LogicDriftAnalyzer(
        original_implementation=black_scholes_call_original,
        refactored_implementation=black_scholes_call_original,
        tolerance_absolute=1.0e-6,
        tolerance_relative=1.0e-5
    )
    
    results = analyzer.run_drift_analysis(n_simulations=1000)
    print(results.get_summary())
    
    # Save results
    analyzer.save_results(results, '/tmp/drift_analysis_test.json')
    print(f"\nResults saved to /tmp/drift_analysis_test.json")
