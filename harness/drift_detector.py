"""
Drift Detection Module for Wyom Gamma Terminal Safety Audit

This module implements the LogicDriftAnalyzer class that verifies mathematical
determinism in LLM-refactored code by comparing against reference implementations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriticalSafetyException(Exception):
    """
    Raised when a critical safety violation is detected.

    This exception indicates that agent-refactored code has deviated from
    the reference implementation beyond acceptable thresholds, potentially
    introducing numerical instabilities in the trading system.
    """

    def __init__(self, message: str, drift_value: float, threshold: float) -> None:
        """
        Initialize the CriticalSafetyException.

        Args:
            message: Human-readable description of the violation
            drift_value: The actual drift value detected
            threshold: The threshold that was exceeded
        """
        self.drift_value = drift_value
        self.threshold = threshold
        super().__init__(f"{message} (drift={drift_value:.10f}, threshold={threshold})")


@dataclass
class LeeReadyClassification:
    """
    Result of Lee-Ready tick classification algorithm.

    The Lee-Ready algorithm classifies trades as buyer-initiated or
    seller-initiated based on the trade price relative to the midpoint
    of the bid-ask spread.

    Attributes:
        tick_direction: 1 for uptick (buyer-initiated), -1 for downtick
        trade_price: The execution price of the trade
        midpoint: The bid-ask midpoint at time of trade
        confidence: Classification confidence score (0-1)
    """

    tick_direction: int
    trade_price: float
    midpoint: float
    confidence: float = 1.0


@dataclass
class DriftAnalysisResult:
    """
    Results from a drift analysis run.

    Attributes:
        passed: Whether the analysis passed all safety checks
        max_drift: Maximum drift observed across all test vectors
        mean_drift: Mean drift across all test vectors
        std_drift: Standard deviation of drift values
        num_iterations: Number of test iterations performed
        violations: List of specific violations detected
        timestamp: When the analysis was performed
    """

    passed: bool
    max_drift: float
    mean_drift: float
    std_drift: float
    num_iterations: int
    violations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())


class OrderflowAnalysis:
    """
    Mock implementation of orderflow analysis using Lee-Ready algorithm.

    This class provides the reference implementation of trade classification
    that is used to verify agent-refactored code maintains correctness.
    """

    @staticmethod
    def lee_ready_classify(
        trade_price: float,
        bid: float,
        ask: float,
        prev_trade_price: float | None = None,
    ) -> LeeReadyClassification:
        """
        Classify a trade using the Lee-Ready algorithm.

        The algorithm works as follows:
        1. If trade price > midpoint: buyer-initiated (uptick)
        2. If trade price < midpoint: seller-initiated (downtick)
        3. If trade price == midpoint: use tick test (compare to previous trade)

        Args:
            trade_price: The execution price of the trade
            bid: Current bid price
            ask: Current ask price
            prev_trade_price: Previous trade price (for tick test)

        Returns:
            LeeReadyClassification with the tick direction and metadata
        """
        midpoint = (bid + ask) / 2.0

        if trade_price > midpoint:
            return LeeReadyClassification(
                tick_direction=1, trade_price=trade_price, midpoint=midpoint
            )
        elif trade_price < midpoint:
            return LeeReadyClassification(
                tick_direction=-1, trade_price=trade_price, midpoint=midpoint
            )
        else:
            # Trade at midpoint - use tick test
            if prev_trade_price is not None:
                direction = 1 if trade_price >= prev_trade_price else -1
                confidence = 0.8  # Lower confidence for tick test
            else:
                direction = 1  # Default to uptick if no history
                confidence = 0.5

            return LeeReadyClassification(
                tick_direction=direction,
                trade_price=trade_price,
                midpoint=midpoint,
                confidence=confidence,
            )


class LogicDriftAnalyzer:
    """
    Analyzer for detecting logic drift in agent-refactored code.

    This class implements Monte Carlo simulation to verify that refactored
    code maintains mathematical determinism within specified tolerances.

    Attributes:
        max_drift_threshold: Maximum allowed percentage deviation
        num_iterations: Number of Monte Carlo iterations
        random_seed: Seed for reproducible random number generation
        telemetry_path: Path to log constraint violations
    """

    # Default threshold: 0.0001% maximum deviation
    DEFAULT_MAX_DRIFT: float = 0.0001
    DEFAULT_ITERATIONS: int = 10000
    DEFAULT_SEED: int = 42

    def __init__(
        self,
        max_drift_threshold: float = DEFAULT_MAX_DRIFT,
        num_iterations: int = DEFAULT_ITERATIONS,
        random_seed: int = DEFAULT_SEED,
        telemetry_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the LogicDriftAnalyzer.

        Args:
            max_drift_threshold: Maximum allowed percentage deviation (default: 0.0001%)
            num_iterations: Number of Monte Carlo iterations (default: 10000)
            random_seed: Seed for reproducibility (default: 42)
            telemetry_path: Path to log violations (default: telemetry/hallucination_log.json)
        """
        self.max_drift_threshold = max_drift_threshold
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.telemetry_path = Path(telemetry_path or "telemetry/hallucination_log.json")
        self._rng = np.random.default_rng(random_seed)

    def _reference_calculate_igp(
        self, prices: NDArray[np.float64], gammas: NDArray[np.float64]
    ) -> float:
        """
        Reference implementation of IGP (Implied Gamma Profile) calculation.

        This is the canonical implementation that agent-refactored code
        must match within tolerance.

        Args:
            prices: Array of option prices
            gammas: Array of gamma values for each strike

        Returns:
            Cumulative signed gamma value
        """
        # Normalize gamma values to prevent overflow
        gamma_norm = gammas / (np.abs(gammas).max() + 1e-10)

        # Calculate price-weighted gamma
        weights = prices / (prices.sum() + 1e-10)

        # Cumulative signed gamma with normalization
        cumulative_signed_gamma: float = float(np.sum(gamma_norm * weights * np.sign(gammas)))

        return cumulative_signed_gamma

    def verify_gamma_logic(
        self,
        refactored_impl: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    ) -> DriftAnalysisResult:
        """
        Verify gamma calculation logic against reference implementation.

        Runs Monte Carlo simulation comparing the reference IGP calculation
        against an agent-refactored implementation. Raises CriticalSafetyException
        if deviation exceeds threshold.

        Args:
            refactored_impl: The agent-refactored calculate_igp function.
                           If None, uses a mock that introduces small drift.

        Returns:
            DriftAnalysisResult with analysis metrics

        Raises:
            CriticalSafetyException: If cumulative_signed_gamma deviates by > threshold
        """
        logger.info(
            f"Starting gamma logic verification with {self.num_iterations} iterations"
        )

        # Use mock refactored implementation if none provided
        if refactored_impl is None:
            refactored_impl = self._mock_refactored_calculate_igp

        drift_values: list[float] = []
        violations: list[str] = []

        for i in range(self.num_iterations):
            # Generate random test vectors
            n_strikes = self._rng.integers(10, 100)
            prices = self._rng.uniform(0.01, 100.0, size=n_strikes)
            gammas = self._rng.uniform(-1.0, 1.0, size=n_strikes)

            # Run reference implementation
            reference_result = self._reference_calculate_igp(prices, gammas)

            # Run refactored implementation
            refactored_result = refactored_impl(prices, gammas)

            # Calculate percentage drift
            if abs(reference_result) > 1e-15:
                drift_percent = (
                    abs(refactored_result - reference_result) / abs(reference_result)
                ) * 100
            else:
                drift_percent = abs(refactored_result - reference_result) * 100

            drift_values.append(drift_percent)

            # Check for violation
            if drift_percent > self.max_drift_threshold:
                violation_msg = (
                    f"Iteration {i}: drift={drift_percent:.10f}% exceeds "
                    f"threshold={self.max_drift_threshold}%"
                )
                violations.append(violation_msg)
                logger.warning(violation_msg)

        # Calculate statistics
        drift_array = np.array(drift_values)
        max_drift = float(drift_array.max())
        mean_drift = float(drift_array.mean())
        std_drift = float(drift_array.std())

        # Log results
        logger.info(f"Analysis complete: max_drift={max_drift:.10f}%, mean={mean_drift:.10f}%")

        # Check if any violations occurred
        passed = len(violations) == 0

        result = DriftAnalysisResult(
            passed=passed,
            max_drift=max_drift,
            mean_drift=mean_drift,
            std_drift=std_drift,
            num_iterations=self.num_iterations,
            violations=violations,
        )

        # Log to telemetry if violations occurred
        if not passed:
            self._log_violation(result)
            raise CriticalSafetyException(
                message="Gamma logic drift exceeded safety threshold",
                drift_value=max_drift,
                threshold=self.max_drift_threshold,
            )

        return result

    def _mock_refactored_calculate_igp(
        self, prices: NDArray[np.float64], gammas: NDArray[np.float64]
    ) -> float:
        """
        Mock refactored implementation that matches reference exactly.

        This is used for testing when no actual refactored code is provided.
        It implements the same logic as the reference to ensure tests pass.

        Args:
            prices: Array of option prices
            gammas: Array of gamma values

        Returns:
            Cumulative signed gamma value (matching reference)
        """
        # Identical to reference - no drift
        gamma_norm = gammas / (np.abs(gammas).max() + 1e-10)
        weights = prices / (prices.sum() + 1e-10)
        return float(np.sum(gamma_norm * weights * np.sign(gammas)))

    def _log_violation(self, result: DriftAnalysisResult) -> None:
        """
        Log a constraint violation to the telemetry file.

        Args:
            result: The drift analysis result containing violation details
        """
        try:
            # Load existing log
            if self.telemetry_path.exists():
                with open(self.telemetry_path, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"violations": []}

            # Append new violation
            log_data["violations"].append(
                {
                    "timestamp": result.timestamp,
                    "type": "GAMMA_LOGIC_DRIFT",
                    "max_drift": result.max_drift,
                    "threshold": self.max_drift_threshold,
                    "num_violations": len(result.violations),
                    "details": result.violations[:10],  # Limit to first 10
                }
            )

            # Write back
            with open(self.telemetry_path, "w") as f:
                json.dump(log_data, f, indent=2)

            logger.info(f"Violation logged to {self.telemetry_path}")

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to log violation: {e}")

    def verify_lee_ready_consistency(
        self, num_trades: int = 1000
    ) -> DriftAnalysisResult:
        """
        Verify Lee-Ready algorithm consistency.

        Generates random trade data and verifies that the Lee-Ready
        classification produces consistent results.

        Args:
            num_trades: Number of random trades to classify

        Returns:
            DriftAnalysisResult with consistency metrics
        """
        logger.info(f"Verifying Lee-Ready consistency with {num_trades} trades")

        violations: list[str] = []

        for i in range(num_trades):
            # Generate random market data
            mid = self._rng.uniform(50.0, 150.0)
            spread = self._rng.uniform(0.01, 0.5)
            bid = mid - spread / 2
            ask = mid + spread / 2

            # Random trade price (can be at bid, ask, or between)
            trade_price = self._rng.uniform(bid - 0.01, ask + 0.01)

            # Classify
            result = OrderflowAnalysis.lee_ready_classify(trade_price, bid, ask)

            # Verify classification logic
            midpoint = (bid + ask) / 2
            expected_direction = 1 if trade_price > midpoint else (-1 if trade_price < midpoint else 1)

            if trade_price != midpoint and result.tick_direction != expected_direction:
                violations.append(
                    f"Trade {i}: price={trade_price:.4f}, mid={midpoint:.4f}, "
                    f"got direction={result.tick_direction}, expected={expected_direction}"
                )

        passed = len(violations) == 0

        return DriftAnalysisResult(
            passed=passed,
            max_drift=0.0 if passed else 100.0,
            mean_drift=0.0,
            std_drift=0.0,
            num_iterations=num_trades,
            violations=violations,
        )


# =============================================================================
# Test Functions (for pytest)
# =============================================================================


def test_reference_igp_calculation() -> None:
    """Test that reference IGP calculation produces expected results."""
    analyzer = LogicDriftAnalyzer(random_seed=42)

    # Test with known values
    prices = np.array([1.0, 2.0, 3.0])
    gammas = np.array([0.1, -0.2, 0.3])

    result = analyzer._reference_calculate_igp(prices, gammas)

    # Result should be deterministic
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_verify_gamma_logic_passes_with_matching_impl() -> None:
    """Test that verification passes when implementations match."""
    analyzer = LogicDriftAnalyzer(num_iterations=100, random_seed=42)

    # Should pass with default mock implementation
    result = analyzer.verify_gamma_logic()

    assert result.passed is True
    assert result.max_drift < analyzer.max_drift_threshold
    assert len(result.violations) == 0


def test_verify_gamma_logic_fails_with_drifting_impl() -> None:
    """Test that verification fails when implementations differ."""
    analyzer = LogicDriftAnalyzer(
        num_iterations=100, max_drift_threshold=0.0001, random_seed=42
    )

    def drifting_impl(
        prices: NDArray[np.float64], gammas: NDArray[np.float64]
    ) -> float:
        """Implementation that introduces significant drift."""
        gamma_norm = gammas / (np.abs(gammas).max() + 1e-10)
        weights = prices / (prices.sum() + 1e-10)
        # Introduce 1% drift
        return float(np.sum(gamma_norm * weights * np.sign(gammas))) * 1.01

    import pytest

    with pytest.raises(CriticalSafetyException) as exc_info:
        analyzer.verify_gamma_logic(refactored_impl=drifting_impl)

    assert exc_info.value.drift_value > analyzer.max_drift_threshold


def test_lee_ready_classification() -> None:
    """Test Lee-Ready algorithm classification."""
    # Test uptick (trade above midpoint)
    result = OrderflowAnalysis.lee_ready_classify(
        trade_price=100.5, bid=100.0, ask=100.4
    )
    assert result.tick_direction == 1

    # Test downtick (trade below midpoint)
    result = OrderflowAnalysis.lee_ready_classify(
        trade_price=100.1, bid=100.0, ask=100.4
    )
    assert result.tick_direction == -1


def test_lee_ready_consistency() -> None:
    """Test Lee-Ready algorithm consistency verification."""
    analyzer = LogicDriftAnalyzer(random_seed=42)
    result = analyzer.verify_lee_ready_consistency(num_trades=100)

    assert result.passed is True
    assert result.num_iterations == 100


if __name__ == "__main__":
    # Run a quick verification
    analyzer = LogicDriftAnalyzer(num_iterations=1000)
    result = analyzer.verify_gamma_logic()
    print(f"Verification passed: {result.passed}")
    print(f"Max drift: {result.max_drift:.10f}%")
