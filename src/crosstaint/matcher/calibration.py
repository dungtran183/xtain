"""Platt calibration for calibrated probability estimates."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


class PlattCalibrator:
    """Platt scaling for calibrated probability estimates.

    Fits sigmoid parameters a and b on validation set to minimize
    negative log-likelihood. The calibrated score is computed as:
        score_calibrated = sigmoid(a * score + b)

    This provides well-calibrated probability estimates that can be
    interpreted as true probabilities of match correctness.
    """

    def __init__(self) -> None:
        """Initialize the Platt calibrator."""
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.fitted: bool = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        verbose: bool = False,
    ) -> None:
        """Fit Platt scaling parameters on validation data.

        Minimizes negative log-likelihood:
            -sum(y * log(p) + (1-y) * log(1-p))
        where p = sigmoid(a * score + b)

        Args:
            scores: Array of raw matcher scores in [0, 1]
            labels: Array of binary labels (0 or 1)
            verbose: If True, print optimization details
        """
        if len(scores) == 0 or len(labels) == 0:
            raise ValueError("Scores and labels arrays cannot be empty")

        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have the same length")

        eps = 1e-7
        scores = np.clip(scores, eps, 1 - eps)

        def negative_log_likelihood(params: np.ndarray) -> float:
            """Compute negative log-likelihood for given parameters."""
            a, b = params
            logits = a * scores + b
            p = 1 / (1 + np.exp(-logits))

            p = np.clip(p, eps, 1 - eps)

            nll = -np.mean(
                labels * np.log(p) + (1 - labels) * np.log(1 - p)
            )
            return nll

        initial_params = np.array([1.0, 0.0])

        bounds = [
            (0.01, 100.0),
            (-10.0, 10.0),
        ]

        result = minimize(
            negative_log_likelihood,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        self.a = float(result.x[0])
        self.b = float(result.x[1])
        self.fitted = True

        if verbose:
            final_nll = negative_log_likelihood(result.x)
            print(f"Platt calibration fit: a={self.a:.4f}, b={self.b:.4f}")
            print(f"Final NLL: {final_nll:.4f}")

    def calibrate(self, score: float) -> float:
        """Calibrate a single score.

        Args:
            score: Raw matcher score in [0, 1]

        Returns:
            Calibrated probability in [0, 1]
        """
        if not self.fitted:
            return score

        eps = 1e-7
        score = np.clip(score, eps, 1 - eps)

        calibrated = 1 / (1 + np.exp(-(self.a * score + self.b)))
        return float(np.clip(calibrated, 0.0, 1.0))

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate a batch of scores.

        Args:
            scores: Array of raw matcher scores

        Returns:
            Array of calibrated probabilities
        """
        return np.array([self.calibrate(s) for s in scores])

    def get_parameters(self) -> tuple[float, float]:
        """Get the fitted Platt scaling parameters.

        Returns:
            Tuple of (a, b) parameters
        """
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted yet")
        return (self.a, self.b)

    def reset(self) -> None:
        """Reset the calibrator to unfitted state."""
        self.a = None
        self.b = None
        self.fitted = False
