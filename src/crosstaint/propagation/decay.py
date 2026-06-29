"""Value decay scheduling and auto-tuning."""

from __future__ import annotations

import bisect
from typing import Callable, Optional

from crosstaint.types import EdgeType


class DecayScheduler:
    """Exponential decay scheduler for taint propagation.

    Computes decayed value as: initial_value * (rho ** hop_distance)

    Args:
        rho: decay rate per hop, in (0, 1]. Default 0.81.
    """

    def __init__(self, rho: float = 0.81) -> None:
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.rho = rho

    def decay(self, hop_distance: int) -> float:
        """Return multiplicative decay factor for a given hop distance."""
        if hop_distance < 0:
            raise ValueError(f"hop_distance must be non-negative, got {hop_distance}")
        if hop_distance == 0:
            return 1.0
        return self.rho ** hop_distance

    def decay_with_value(
        self,
        value: int,
        hop: int,
        initial_value: int,
    ) -> int:
        """Return remaining value after `hop` hops given an initial value.

        Args:
            value: current value (ignored; recomputed from initial_value)
            hop: hop distance
            initial_value: starting value before any decay

        Returns:
            int(initial_value * rho ** hop), floored at 0
        """
        decayed = initial_value * (self.rho ** hop)
        return max(0, int(decayed))


class AutoTuner:
    """Automatically tunes rho to achieve a target false-positive rate.

    Uses bisection search over rho_range, evaluating the actual FPR from
    the provided eval_fn at each candidate rho. Stops when |FPR - target|
    is below tolerance or max_steps is reached.

    Args:
        rho_range: tuple of (min_rho, max_rho) search bounds.
        target_fp_rate: desired false-positive rate (default 0.05).
        max_steps: maximum bisection iterations (default 8).
        eval_fn: callable taking rho and returning the observed FPR.
                 Must be provided unless tune() is called with an override.
    """

    def __init__(
        self,
        rho_range: tuple[float, float] = (0.70, 0.90),
        target_fp_rate: float = 0.05,
        max_steps: int = 8,
        eval_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.rho_range = rho_range
        self.target_fp_rate = target_fp_rate
        self.max_steps = max_steps
        self._eval_fn = eval_fn

        self._current_rho = (rho_range[0] + rho_range[1]) / 2.0
        self._best_rho = self._current_rho
        self._best_diff = float("inf")

    def suggest_rho(self) -> float:
        """Return the current best rho found so far."""
        return self._current_rho

    def tune(
        self,
        eval_fn: Optional[Callable[[float], float]] = None,
    ) -> float:
        """Run bisection search to find optimal rho.

        Args:
            eval_fn: optional override for the evaluation function.

        Returns:
            optimal rho value.
        """
        fn = eval_fn if eval_fn is not None else self._eval_fn
        if fn is None:
            raise ValueError("eval_fn must be provided either at init or at tune()")

        lo, hi = self.rho_range

        for _ in range(self.max_steps):
            mid = (lo + hi) / 2.0
            fpr = fn(mid)
            diff = abs(fpr - self.target_fp_rate)

            if diff < 0.01:
                self._current_rho = mid
                self._best_rho = mid
                self._best_diff = diff
                return mid

            if fpr > self.target_fp_rate:
                hi = mid
            else:
                lo = mid

            if diff < self._best_diff:
                self._best_diff = diff
                self._best_rho = mid

        self._current_rho = self._best_rho
        return self._best_rho
