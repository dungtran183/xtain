"""Bridge event matcher package for CrossTaint."""

from crosstaint.matcher.calibration import PlattCalibrator
from crosstaint.matcher.featurizer import EventFeaturizer
from crosstaint.matcher.model import (
    BilinearBaseline,
    BilinearScoringHead,
    BridgeEventMatcher,
    EventEmbedding,
    MatcherOutput,
    PositionalEncoding,
    SiameseMLPMatcher,
)
from crosstaint.matcher.trainer import HardNegativeMiner, MatcherTrainer, TaintPairDataset

__all__ = [
    "BridgeEventMatcher",
    "EventFeaturizer",
    "MatcherTrainer",
    "PlattCalibrator",
    "BilinearBaseline",
    "SiameseMLPMatcher",
    "PositionalEncoding",
    "EventEmbedding",
    "BilinearScoringHead",
    "MatcherOutput",
    "TaintPairDataset",
    "HardNegativeMiner",
]
