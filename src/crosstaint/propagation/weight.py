"""Edge weights for taint propagation.

Modified in derived release v1.1.0 so fixed-weight propagation can be checked
without importing the optional trainable PyTorch path.
"""

from __future__ import annotations

from typing import Mapping

from crosstaint.types import EdgeType, IREdge


try:
    import torch
    import torch.nn as nn
except ImportError:  # Fixed-weight propagation does not require PyTorch.
    torch = None
    nn = None


_EDGE_TYPE_ALIASES: dict[str, str] = {
    "cross_chain_bridge": EdgeType.CROSS_CHAIN_BRIDGE,
    "intra_chain_transfer": EdgeType.INTRA_CHAIN_TRANSFER,
    "dex_swap": EdgeType.DEX_SWAP,
}


class EdgeWeightLearner:
    DEFAULT_WEIGHTS: dict[str, float] = {
        EdgeType.INTRA_CHAIN_TRANSFER: 1.00,
        EdgeType.CROSS_CHAIN_BRIDGE: 0.90,
        EdgeType.DEX_SWAP: 0.80,
    }

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        trainable: bool = False,
    ) -> None:
        self.trainable = trainable
        self._weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for key, value in weights.items():
                edge_type = _EDGE_TYPE_ALIASES.get(key.lower(), key)
                self._weights[edge_type] = float(value)

        if trainable:
            if torch is None or nn is None:
                raise RuntimeError(
                    "trainable edge weights require PyTorch; install the full runtime dependencies"
                )
            self._mlp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
            base = torch.tensor(
                [self._weights.get(EdgeType.INTRA_CHAIN_TRANSFER, 1.0)],
                dtype=torch.float32,
            )
            with torch.no_grad():
                self._mlp[2].bias.fill_(torch.log(base / (1 - base + 1e-8)).item())
        else:
            self._mlp = None

    @classmethod
    def from_config(cls, propagation_cfg: Mapping[str, object]) -> "EdgeWeightLearner":
        edge_weights = propagation_cfg.get("edge_weights", {})
        if isinstance(edge_weights, dict):
            return cls(weights=edge_weights)
        return cls()

    def get_weight(self, edge: IREdge) -> float:
        return self.get_weight_by_type(edge.edge_type)

    def get_weight_by_type(self, edge_type: str) -> float:
        if self._mlp is not None and self.trainable:
            assert torch is not None
            base = self._weights.get(edge_type, 0.9)
            weight = self._mlp(torch.tensor([[base]]))
            return float(weight.item())
        return self._weights.get(edge_type, 0.9)
