"""Transformer-based bridge event matcher for CrossTaint."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from crosstaint.types import EventFeatures


@dataclass
class MatcherOutput:
    """Output from the bridge event matcher."""

    score: torch.Tensor
    embedding1: Optional[torch.Tensor] = None
    embedding2: Optional[torch.Tensor] = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input.

    Generates positional embeddings using sine and cosine functions
    of different frequencies, following 'Attention is All You Need'.
    """

    def __init__(self, d_model: int = 128, max_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class EventEmbedding(nn.Module):
    """Embeds 5-token event tuple into 128-dimensional space.

    The 5-token event tuple consists of:
    - selector_embedding (4-dim): hashed to bucket 0-3
    - topics_embedding (64-dim): hash projection with Tanh activation
    - value_bracket (1-dim): log10 discretized value range
    - timestamp_feature (1-dim): hour-of-week normalized to [0, 1]
    - context_embedding (32-dim): bridge-family lookup table

    Total: 4 + 64 + 1 + 1 + 32 = 102 dimensions
    Final output is projected to 128 dimensions.
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

        self.selector_proj = nn.Linear(4, 16)
        self.topics_proj = nn.Linear(64, 64)
        self.value_proj = nn.Linear(1, 8)
        self.timestamp_proj = nn.Linear(1, 8)
        self.context_proj = nn.Linear(32, 16)

        total_input_dim = 16 + 64 + 8 + 8 + 16
        self.output_proj = nn.Linear(total_input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selector = x[:, :, :4]
        topics = x[:, :, 4:68]
        value = x[:, :, 68:69]
        timestamp = x[:, :, 69:70]
        context = x[:, :, 70:102]

        selector_emb = torch.tanh(self.selector_proj(selector))
        topics_emb = torch.tanh(self.topics_proj(topics))
        value_emb = torch.tanh(self.value_proj(value))
        timestamp_emb = torch.tanh(self.timestamp_proj(timestamp))
        context_emb = torch.tanh(self.context_proj(context))

        combined = torch.cat(
            [selector_emb, topics_emb, value_emb, timestamp_emb, context_emb],
            dim=-1
        )

        embedded = self.output_proj(combined)
        return self.layer_norm(embedded)


class BilinearScoringHead(nn.Module):
    """Bilinear scoring head for computing similarity between event embeddings.

    Takes two 128-dim embeddings and computes a similarity score using
    a learned bilinear transformation: score = sigmoid(w * (e1 * W * e2 + b))
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.w = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor
    ) -> torch.Tensor:
        # e1, e2 shape: [batch, d_model]
        bilinear = e1 * self.W(e2)
        bilinear = bilinear.sum(dim=-1, keepdim=True)
        score = self.w(bilinear)
        return self.sigmoid(score).squeeze(-1)


class BridgeEventMatcher(nn.Module):
    """Siamese network for matching bridge events across chains.

    Uses a transformer encoder architecture to process event sequences
    and compute similarity scores between event pairs.

    Architecture:
    - EventEmbedding: projects raw features to d_model dimensions
    - PositionalEncoding: adds positional information
    - 4 TransformerEncoderLayers with 4 attention heads each
    - BilinearScoringHead: computes final similarity score
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.event_embedding = EventEmbedding(d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.scoring_head = BilinearScoringHead(d_model)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        return_embeddings: bool = False,
    ) -> MatcherOutput:
        """Forward pass computing similarity between two event sequences.

        Args:
            e1: First event tensor of shape [batch, 5, 102]
            e2: Second event tensor of shape [batch, 5, 102]
            return_embeddings: If True, include embeddings in output

        Returns:
            MatcherOutput with score in [0, 1] and optionally embeddings
        """
        emb1 = self.event_embedding(e1)
        emb2 = self.event_embedding(e2)

        emb1 = self.positional_encoding(emb1)
        emb2 = self.positional_encoding(emb2)

        trans1 = self.transformer(emb1)
        trans2 = self.transformer(emb2)

        pooled1 = trans1.mean(dim=1)
        pooled2 = trans2.mean(dim=1)

        score = self.scoring_head(pooled1, pooled2)

        if return_embeddings:
            return MatcherOutput(
                score=score,
                embedding1=pooled1,
                embedding2=pooled2,
            )
        return MatcherOutput(score=score)


class BilinearBaseline(nn.Module):
    """Bilinear baseline model without transformer for ablation studies.

    Simpler architecture that directly embeds events and computes
    bilinear similarity without transformer processing.
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

        self.event_embedding = EventEmbedding(d_model)
        self.scoring_head = BilinearScoringHead(d_model)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        return_embeddings: bool = False,
    ) -> MatcherOutput:
        """Forward pass without transformer layers.

        Args:
            e1: First event tensor of shape [batch, 5, 102]
            e2: Second event tensor of shape [batch, 5, 102]
            return_embeddings: If True, include embeddings in output

        Returns:
            MatcherOutput with score in [0, 1] and optionally embeddings
        """
        emb1 = self.event_embedding(e1)
        emb2 = self.event_embedding(e2)

        pooled1 = emb1.mean(dim=1)
        pooled2 = emb2.mean(dim=1)

        score = self.scoring_head(pooled1, pooled2)

        if return_embeddings:
            return MatcherOutput(
                score=score,
                embedding1=pooled1,
                embedding2=pooled2,
            )
        return MatcherOutput(score=score)


class SiameseMLPMatcher(nn.Module):
    """3-layer MLP baseline for siamese network comparison.

    Baseline model that uses MLPs to process each event branch
    and computes similarity via concatenated features.
    """

    def __init__(
        self,
        input_dim: int = 102,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        return_embeddings: bool = False,
    ) -> MatcherOutput:
        """Forward pass with MLP branches.

        Args:
            e1: First event tensor of shape [batch, 5, 102]
            e2: Second event tensor of shape [batch, 5, 102]
            return_embeddings: If True, include embeddings in output

        Returns:
            MatcherOutput with score in [0, 1] and optionally embeddings
        """
        emb1_flat = e1.mean(dim=1)
        emb2_flat = e2.mean(dim=1)

        feat1 = self.branch(emb1_flat)
        feat2 = self.branch(emb2_flat)

        concatenated = torch.cat([feat1, feat2], dim=-1)
        score = self.fusion(concatenated).squeeze(-1)

        if return_embeddings:
            return MatcherOutput(
                score=score,
                embedding1=feat1,
                embedding2=feat2,
            )
        return MatcherOutput(score=score)
