"""
Latency Stress Test Module for Wyom Gamma Terminal Safety Audit

This module measures whether agent-refactored code introduces latency
that exceeds acceptable thresholds for high-frequency trading systems.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the IBKR message loop."""

    TICK_PRICE = "TICK_PRICE"
    TICK_SIZE = "TICK_SIZE"
    ORDER_STATUS = "ORDER_STATUS"
    EXECUTION = "EXECUTION"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    MARKET_DATA = "MARKET_DATA"
    OPTION_CHAIN = "OPTION_CHAIN"


@dataclass
class IBKRMessage:
    """
    Mock IBKR message structure.

    Attributes:
        msg_type: Type of the message
        ticker_id: Identifier for the ticker/contract
        data: Message payload
        timestamp: When the message was created
        sequence_num: Message sequence number for ordering
    """

    msg_type: MessageType
    ticker_id: int
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0


@dataclass
class LatencyResult:
    """
    Results from a latency measurement.

    Attributes:
        passed: Whether latency was within acceptable bounds
        mean_latency_ms: Mean processing latency in milliseconds
        max_latency_ms: Maximum observed latency
        min_latency_ms: Minimum observed latency
        p99_latency_ms: 99th percentile latency
        p95_latency_ms: 95th percentile latency
        std_latency_ms: Standard deviation of latency
        num_batches: Number of batches processed
        threshold_ms: The latency threshold used
        violations: List of batches that exceeded threshold
    """

    passed: bool
    mean_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    p99_latency_ms: float
    p95_latency_ms: float
    std_latency_ms: float
    num_batches: int
    threshold_ms: float
    violations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())


class MockIBKRMessageLoop:
    """
    Mock implementation of the IBKR message loop.

    This simulates the Interactive Brokers message queue for testing
    latency characteristics of the strategy orchestrator.

    Attributes:
        batch_size: Number of messages per batch
        message_rate_hz: Simulated message arrival rate
    """

    DEFAULT_BATCH_SIZE: int = 100
    DEFAULT_MESSAGE_RATE: int = 10000  # 10k messages/second

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        message_rate_hz: int = DEFAULT_MESSAGE_RATE,
    ) -> None:
        """
        Initialize the mock IBKR message loop.

        Args:
            batch_size: Number of messages per batch
            message_rate_hz: Simulated message rate in Hz
        """
        self.batch_size = batch_size
        self.message_rate_hz = message_rate_hz
        self._sequence_counter = 0
        self._rng = np.random.default_rng(42)

    def generate_message_batch(self) -> list[IBKRMessage]:
        """
        Generate a batch of mock IBKR messages.

        Returns:
            List of IBKRMessage objects simulating market data
        """
        messages: list[IBKRMessage] = []

        for _ in range(self.batch_size):
            msg_type = self._rng.choice(list(MessageType))
            ticker_id = int(self._rng.integers(1000, 9999))

            # Generate realistic market data
            base_price = float(self._rng.uniform(10.0, 500.0))
            data = {
                "price": base_price,
                "size": int(self._rng.integers(1, 10000)),
                "bid": base_price - float(self._rng.uniform(0.01, 0.10)),
                "ask": base_price + float(self._rng.uniform(0.01, 0.10)),
                "volume": int(self._rng.integers(10000, 1000000)),
            }

            self._sequence_counter += 1

            messages.append(
                IBKRMessage(
                    msg_type=msg_type,
                    ticker_id=ticker_id,
                    data=data,
                    sequence_num=self._sequence_counter,
                )
            )

        return messages


class StrategyOrchestrator:
    """
    Mock strategy orchestrator for latency testing.

    This represents the component that processes IBKR messages
    and executes trading strategies.

    Attributes:
        processing_delay_us: Simulated processing delay in microseconds
    """

    DEFAULT_PROCESSING_DELAY: int = 100  # 100 microseconds base delay

    def __init__(self, processing_delay_us: int = DEFAULT_PROCESSING_DELAY) -> None:
        """
        Initialize the strategy orchestrator.

        Args:
            processing_delay_us: Base processing delay in microseconds
        """
        self.processing_delay_us = processing_delay_us
        self._processed_count = 0
        self._gamma_accumulator = 0.0

    def process_batch(self, messages: list[IBKRMessage]) -> dict[str, Any]:
        """
        Process a batch of IBKR messages.

        This method simulates the actual processing that would occur
        in the real strategy orchestrator, including gamma calculations.

        Args:
            messages: List of IBKR messages to process

        Returns:
            Dictionary with processing results and metrics
        """
        start_time = time.perf_counter()

        # Simulate processing each message
        for msg in messages:
            self._process_single_message(msg)

        # Simulate additional batch-level processing
        self._run_gamma_calculations(len(messages))

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        self._processed_count += len(messages)

        return {
            "batch_size": len(messages),
            "processing_time_ms": processing_time_ms,
            "total_processed": self._processed_count,
            "gamma_accumulator": self._gamma_accumulator,
        }

    def _process_single_message(self, msg: IBKRMessage) -> None:
        """
        Process a single IBKR message.

        Args:
            msg: The message to process
        """
        # Simulate message-level processing
        _ = msg.data.get("price", 0) * msg.data.get("size", 0)

        # Simulate small processing delay (CPU-bound work)
        # Using a tight loop instead of sleep for more realistic latency
        iterations = self.processing_delay_us // 10
        _sum = 0.0
        for i in range(iterations):
            _sum += i * 0.001

    def _run_gamma_calculations(self, batch_size: int) -> None:
        """
        Simulate batch-level gamma calculations.

        Args:
            batch_size: Number of messages in the batch
        """
        # Simulate gamma exposure calculation
        random_gammas = np.random.randn(batch_size)
        self._gamma_accumulator += float(np.sum(random_gammas))

    async def process_batch_async(
        self, messages: list[IBKRMessage]
    ) -> dict[str, Any]:
        """
        Async version of batch processing.

        Args:
            messages: List of IBKR messages to process

        Returns:
            Dictionary with processing results
        """
        # Run synchronous processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_batch, messages)


class LatencyStressTester:
    """
    Stress tester for measuring latency in the trading system.

    This class generates high-frequency message traffic and measures
    how quickly the strategy orchestrator can process batches.

    Attributes:
        threshold_ms: Maximum acceptable batch processing time
        num_batches: Number of batches to test
    """

    DEFAULT_THRESHOLD_MS: float = 100.0
    DEFAULT_NUM_BATCHES: int = 1000

    def __init__(
        self,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        num_batches: int = DEFAULT_NUM_BATCHES,
    ) -> None:
        """
        Initialize the latency stress tester.

        Args:
            threshold_ms: Maximum acceptable latency in milliseconds
            num_batches: Number of batches to test
        """
        self.threshold_ms = threshold_ms
        self.num_batches = num_batches
        self.message_loop = MockIBKRMessageLoop()
        self.orchestrator = StrategyOrchestrator()

    def run_stress_test(self) -> LatencyResult:
        """
        Run the latency stress test.

        Generates message batches and measures processing latency,
        asserting that all batches are processed within the threshold.

        Returns:
            LatencyResult with detailed metrics
        """
        logger.info(
            f"Starting latency stress test: {self.num_batches} batches, "
            f"threshold={self.threshold_ms}ms"
        )

        latencies: list[float] = []
        violations: list[str] = []

        for i in range(self.num_batches):
            # Generate message batch
            batch = self.message_loop.generate_message_batch()

            # Time the processing
            start_time = time.perf_counter()
            result = self.orchestrator.process_batch(batch)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Check for violation
            if latency_ms > self.threshold_ms:
                violation_msg = (
                    f"Batch {i}: latency={latency_ms:.2f}ms exceeds "
                    f"threshold={self.threshold_ms}ms"
                )
                violations.append(violation_msg)
                logger.warning(violation_msg)

            # Log progress periodically
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Processed {i + 1}/{self.num_batches} batches, "
                    f"current latency={latency_ms:.2f}ms"
                )

        # Calculate statistics
        sorted_latencies = sorted(latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_idx = int(len(sorted_latencies) * 0.95)

        result = LatencyResult(
            passed=len(violations) == 0,
            mean_latency_ms=statistics.mean(latencies),
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            p99_latency_ms=sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else max(latencies),
            p95_latency_ms=sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else max(latencies),
            std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            num_batches=self.num_batches,
            threshold_ms=self.threshold_ms,
            violations=violations,
        )

        logger.info(
            f"Stress test complete: mean={result.mean_latency_ms:.2f}ms, "
            f"max={result.max_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms"
        )

        return result

    async def run_stress_test_async(self) -> LatencyResult:
        """
        Async version of the stress test.

        Returns:
            LatencyResult with detailed metrics
        """
        logger.info(
            f"Starting async latency stress test: {self.num_batches} batches"
        )

        latencies: list[float] = []
        violations: list[str] = []

        for i in range(self.num_batches):
            batch = self.message_loop.generate_message_batch()

            start_time = time.perf_counter()
            await self.orchestrator.process_batch_async(batch)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if latency_ms > self.threshold_ms:
                violations.append(
                    f"Batch {i}: latency={latency_ms:.2f}ms exceeds threshold"
                )

        sorted_latencies = sorted(latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_idx = int(len(sorted_latencies) * 0.95)

        return LatencyResult(
            passed=len(violations) == 0,
            mean_latency_ms=statistics.mean(latencies),
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            p99_latency_ms=sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else max(latencies),
            p95_latency_ms=sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else max(latencies),
            std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            num_batches=self.num_batches,
            threshold_ms=self.threshold_ms,
            violations=violations,
        )


def measure_refactored_latency(
    reference_orchestrator: StrategyOrchestrator,
    refactored_orchestrator: StrategyOrchestrator,
    num_batches: int = 100,
    max_latency_added_ms: float = 5.0,
) -> tuple[bool, float]:
    """
    Compare latency between reference and refactored implementations.

    Args:
        reference_orchestrator: The original implementation
        refactored_orchestrator: The agent-refactored implementation
        num_batches: Number of batches to compare
        max_latency_added_ms: Maximum acceptable latency increase

    Returns:
        Tuple of (passed, latency_diff_ms)
    """
    message_loop = MockIBKRMessageLoop()

    reference_latencies: list[float] = []
    refactored_latencies: list[float] = []

    for _ in range(num_batches):
        batch = message_loop.generate_message_batch()

        # Measure reference
        start = time.perf_counter()
        reference_orchestrator.process_batch(copy.deepcopy(batch))
        reference_latencies.append((time.perf_counter() - start) * 1000)

        # Measure refactored
        start = time.perf_counter()
        refactored_orchestrator.process_batch(copy.deepcopy(batch))
        refactored_latencies.append((time.perf_counter() - start) * 1000)

    mean_reference = statistics.mean(reference_latencies)
    mean_refactored = statistics.mean(refactored_latencies)
    latency_diff = mean_refactored - mean_reference

    passed = latency_diff <= max_latency_added_ms

    logger.info(
        f"Latency comparison: reference={mean_reference:.2f}ms, "
        f"refactored={mean_refactored:.2f}ms, diff={latency_diff:.2f}ms"
    )

    return passed, latency_diff


# =============================================================================
# Test Functions (for pytest)
# =============================================================================


def test_ibkr_message_loop_generates_batches() -> None:
    """Test that the mock message loop generates valid batches."""
    loop = MockIBKRMessageLoop(batch_size=50)
    batch = loop.generate_message_batch()

    assert len(batch) == 50
    assert all(isinstance(msg, IBKRMessage) for msg in batch)
    assert all(msg.sequence_num > 0 for msg in batch)


def test_strategy_orchestrator_processes_batch() -> None:
    """Test that the strategy orchestrator processes batches correctly."""
    loop = MockIBKRMessageLoop(batch_size=10)
    orchestrator = StrategyOrchestrator(processing_delay_us=10)

    batch = loop.generate_message_batch()
    result = orchestrator.process_batch(batch)

    assert result["batch_size"] == 10
    assert result["processing_time_ms"] >= 0
    assert result["total_processed"] == 10


def test_latency_stress_test_passes_with_fast_processing() -> None:
    """Test that stress test passes with fast processing."""
    tester = LatencyStressTester(
        threshold_ms=100.0,  # 100ms threshold
        num_batches=10,  # Small number for test speed
    )
    tester.orchestrator.processing_delay_us = 10  # Very fast

    result = tester.run_stress_test()

    assert result.passed is True
    assert result.max_latency_ms < 100.0
    assert len(result.violations) == 0


def test_latency_statistics_calculation() -> None:
    """Test that latency statistics are calculated correctly."""
    tester = LatencyStressTester(num_batches=50)
    tester.orchestrator.processing_delay_us = 10

    result = tester.run_stress_test()

    assert result.mean_latency_ms > 0
    assert result.max_latency_ms >= result.mean_latency_ms
    assert result.min_latency_ms <= result.mean_latency_ms
    assert result.p99_latency_ms >= result.p95_latency_ms


def test_refactored_latency_comparison() -> None:
    """Test latency comparison between reference and refactored implementations."""
    ref_orchestrator = StrategyOrchestrator(processing_delay_us=10)
    refactored_orchestrator = StrategyOrchestrator(processing_delay_us=10)

    passed, diff = measure_refactored_latency(
        ref_orchestrator, refactored_orchestrator, num_batches=10, max_latency_added_ms=5.0
    )

    # Same implementation should have minimal difference
    assert abs(diff) < 5.0


def test_message_types_coverage() -> None:
    """Test that all message types can be generated and processed."""
    loop = MockIBKRMessageLoop(batch_size=1000)
    batch = loop.generate_message_batch()

    # Check that we have variety in message types
    msg_types = {msg.msg_type for msg in batch}
    assert len(msg_types) >= 3  # Should have at least 3 different types


if __name__ == "__main__":
    # Run a quick stress test
    tester = LatencyStressTester(threshold_ms=100.0, num_batches=100)
    result = tester.run_stress_test()

    print(f"\nStress Test Results:")
    print(f"  Passed: {result.passed}")
    print(f"  Mean Latency: {result.mean_latency_ms:.2f}ms")
    print(f"  Max Latency: {result.max_latency_ms:.2f}ms")
    print(f"  P99 Latency: {result.p99_latency_ms:.2f}ms")
    print(f"  Violations: {len(result.violations)}")
