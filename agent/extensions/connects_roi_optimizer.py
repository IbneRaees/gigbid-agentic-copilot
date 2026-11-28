"""Connects ROI Optimizer Extension

This module provides a strongly‑typed, production‑grade extension for the GigBid
agent that estimates the expected return on investment (ROI) of a proposed
"connects" (up‑sell) feature in a bid. It can be plugged into the LangGraph
workflow via the `Extension` base class (defined elsewhere in the codebase).

The optimizer works in three stages:

1. **Probability modelling** – a Bayesian estimator predicts the win‑probability
   of the bid given the presence of a connects offer.
2. **Value calculation** – the expected monetary value of the connects is
   derived from the base prize, the connects multiplier and the probability
   from step 1.
3. **Decision heuristic** – compares the expected value against a configurable
   risk threshold and returns a ``BidDecision`` that downstream nodes can use.

All public symbols are exported via ``__all__`` so that the extension can be
imported dynamically by the agent runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic models for configuration and output
# ---------------------------------------------------------------------------


class ROIConfig(BaseModel):
    """Configuration for the ROI optimizer."""

    base_prize: float = Field(default=1000.0, gt=0)
    connects_multiplier: float = Field(default=1.2, gt=0)
    risk_aversion: float = Field(default=0.7, gt=0, le=1)
    prior_win_prob: float = Field(default=0.4, gt=0, lt=1)

    @field_validator("risk_aversion")
    @classmethod
    def _check_risk(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("risk_aversion must be in (0, 1]")
        return v

    @field_validator("prior_win_prob")
    @classmethod
    def _check_prior(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("prior_win_prob must be in (0, 1)")
        return v


class ROIResult(TypedDict, total=False):
    """Result of a single ROI evaluation.

    Keys
    ----
    win_probability: float
    expected_value: float
    decision: str  # "accept" or "reject"
    """


# ---------------------------------------------------------------------------
# Core optimizer implementation
# ---------------------------------------------------------------------------


class ConnectsROIOptimizer:
    """Calculate ROI for a connects offer.

    The class is deliberately pure‑functional – it holds no mutable state so it can
    be safely instantiated once and reused across many bid evaluations.
    """

    _DEFAULT_CONFIG: Final[ROIConfig] = ROIConfig(
        base_prize=1000.0,
        connects_multiplier=1.2,
        risk_aversion=0.7,
        prior_win_prob=0.4,
    )

    def __init__(self, config: ROIConfig | None = None) -> None:
        self.config = config or self._DEFAULT_CONFIG

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def evaluate(self, *, base_prize: float | None = None, connects: bool = True) -> ROIResult:
        """Compute the ROI and a simple decision.

        Parameters
        ----------
        base_prize: float | None
            Override the ``base_prize`` from the config. If ``None`` the config
            value is used.
        connects: bool
            Whether a connects offer is being considered.

        Returns
        -------
        ROIResult
            Mapping containing ``win_probability``, ``expected_value`` and a
            ``decision`` string.
        """
        cfg = self.config
        prize = base_prize if base_prize is not None else cfg.base_prize

        win_prob = self._posterior_win_probability(prior=cfg.prior_win_prob, has_connects=connects)
        expected = win_prob * (prize * (cfg.connects_multiplier if connects else 1.0))
        min_risk_adjusted_ev = 250.0
        decision = (
            "accept"
            if (connects and expected * cfg.risk_aversion >= min_risk_adjusted_ev)
            else "reject"
        )
        return ROIResult(
            win_probability=win_prob,
            expected_value=expected,
            decision=decision,
        )

    # -------------------------------------------------------------------
    # Internal helpers – kept private to make unit‑testing straightforward
    # -------------------------------------------------------------------
    def _posterior_win_probability(self, *, prior: float, has_connects: bool) -> float:
        """Bayesian update of win probability.

        A very simple model: adding connects increases the odds by a factor of
        ``1.5``. The odds are ``p/(1-p)``; we multiply by the factor and convert
        back to probability.
        """
        if not has_connects:
            return prior
        odds = prior / (1 - prior)
        updated_odds = odds * 1.5
        posterior = updated_odds / (1 + updated_odds)
        # Clamp to a sensible range to avoid numerical extremes.
        return max(0.0, min(1.0, posterior))

    # -------------------------------------------------------------------
    # Extension hook – the agent runtime expects a ``run`` callable.
    # -------------------------------------------------------------------
    def run(self, payload: Mapping[str, object]) -> ROIResult:
        """Entry point used by the extension loader.

        ``payload`` is expected to contain ``base_prize`` (float) and
        ``connects`` (bool). Extra keys are ignored.
        """
        base_prize = float(payload.get("base_prize", self.config.base_prize))
        connects = bool(payload.get("connects", True))
        return self.evaluate(base_prize=base_prize, connects=connects)


# Exported symbol for dynamic loading
__all__ = ["ConnectsROIOptimizer", "ROIConfig", "ROIResult"]
