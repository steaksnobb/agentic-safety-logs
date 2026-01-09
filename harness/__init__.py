"""
Harness package for Wyom Gamma Terminal Safety Audit.

This package contains the core testing components for verifying
that AI-assisted code refactoring maintains mathematical correctness
and performance characteristics.
"""

from harness.drift_detector import (
    CriticalSafetyException,
    DriftAnalysisResult,
    LeeReadyClassification,
    LogicDriftAnalyzer,
    OrderflowAnalysis,
)
from harness.latency_stress_test import (
    IBKRMessage,
    LatencyResult,
    LatencyStressTester,
    MessageType,
    MockIBKRMessageLoop,
    StrategyOrchestrator,
)

__all__ = [
    # From drift_detector
    "CriticalSafetyException",
    "DriftAnalysisResult",
    "LeeReadyClassification",
    "LogicDriftAnalyzer",
    "OrderflowAnalysis",
    # From latency_stress_test
    "IBKRMessage",
    "LatencyResult",
    "LatencyStressTester",
    "MessageType",
    "MockIBKRMessageLoop",
    "StrategyOrchestrator",
]
