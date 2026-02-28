"""
CPM bidding and ranking module for CTV and In-App video.

All billing is CPM-based. Ranking strategies optimize for
video-specific metrics: VTR (view-through rate), completion, engagement.
"""

from __future__ import annotations

from enum import IntEnum

from liteads.common.logger import get_logger
from liteads.schemas.internal import AdCandidate

logger = get_logger(__name__)


class RankingStrategy(IntEnum):
    """Ranking strategy enum for CPM video."""

    CPM = 1  # Pure CPM ranking (bid = eCPM)
    VTR_WEIGHTED = 2  # CPM weighted by predicted VTR
    ENGAGEMENT = 3  # CPM weighted by engagement (CTR + VTR)
    COMPLETION = 4  # CPM weighted by predicted completion rate


class Bidding:
    """
    CPM bidding and ranking calculator for video ads.

    Since all campaigns are CPM, eCPM equals the bid directly.
    Ranking strategies layer video-specific quality signals on top.
    """

    def __init__(
        self,
        strategy: RankingStrategy = RankingStrategy.CPM,
        min_cpm: float = 0.01,
    ):
        self.strategy = strategy
        self.min_cpm = min_cpm

    def calculate_ecpm(self, candidate: AdCandidate) -> float:
        """Calculate eCPM for CPM candidate.

        For CPM billing, eCPM is simply the bid amount.
        """
        return max(candidate.bid, self.min_cpm)

    def calculate_score(self, candidate: AdCandidate) -> float:
        """
        Calculate ranking score based on strategy.

        Strategies:
        - CPM: Pure bid-based (eCPM = bid)
        - VTR_WEIGHTED: eCPM * predicted VTR
        - ENGAGEMENT: eCPM * (VTR + CTR blend)
        - COMPLETION: eCPM * predicted completion rate
        """
        ecpm = self.calculate_ecpm(candidate)
        pvtr = getattr(candidate, "pvtr", None) or 0.70
        pctr = candidate.pctr

        if self.strategy == RankingStrategy.CPM:
            score = ecpm

        elif self.strategy == RankingStrategy.VTR_WEIGHTED:
            # Weight CPM by predicted view-through rate
            vtr_factor = max(pvtr, 0.01)
            score = ecpm * vtr_factor

        elif self.strategy == RankingStrategy.ENGAGEMENT:
            # Blend VTR and CTR signals
            engagement = 0.7 * pvtr + 0.3 * min(pctr * 100, 1.0)
            score = ecpm * max(engagement, 0.01)

        elif self.strategy == RankingStrategy.COMPLETION:
            # Prioritize ads with high completion rate
            completion_factor = max(pvtr, 0.01) ** 0.5
            score = ecpm * completion_factor

        else:
            score = ecpm

        return score

    def rank(
        self,
        candidates: list[AdCandidate],
        apply_ecpm: bool = True,
    ) -> list[AdCandidate]:
        """
        Rank candidates by score (highest first).

        Args:
            candidates: List of candidates to rank
            apply_ecpm: Whether to calculate and apply eCPM

        Returns:
            Sorted list of candidates (highest score first)
        """
        if not candidates:
            return []

        for candidate in candidates:
            if apply_ecpm:
                candidate.ecpm = self.calculate_ecpm(candidate)
            candidate.score = self.calculate_score(candidate)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

        logger.debug(
            f"Ranked {len(ranked)} CPM video candidates",
            top_score=ranked[0].score if ranked else 0,
        )

        return ranked


class SecondPriceAuction:
    """
    Second price auction for CPM video inventory.

    Winner pays the second highest CPM bid (plus small increment).
    """

    def __init__(self, increment: float = 0.01):
        self.increment = increment

    def run_auction(
        self,
        candidates: list[AdCandidate],
    ) -> tuple[AdCandidate | None, float]:
        """
        Run second price auction.

        Returns:
            Tuple of (winner, clearing_cpm)
        """
        if not candidates:
            return None, 0.0

        if len(candidates) == 1:
            winner = candidates[0]
            return winner, self.increment

        winner = candidates[0]
        second_price = candidates[1].ecpm
        price = second_price + self.increment

        return winner, price


class BudgetPacing:
    """
    Budget pacing for smooth ad delivery.

    Ensures budget is spent evenly throughout the day.
    """

    def __init__(
        self,
        daily_budget: float,
        hours_remaining: int = 24,
        smoothing_factor: float = 1.2,
    ):
        """
        Initialize budget pacing.

        Args:
            daily_budget: Total daily budget
            hours_remaining: Hours remaining in the day
            smoothing_factor: Factor to adjust pacing (>1 = aggressive)
        """
        self.daily_budget = daily_budget
        self.hours_remaining = max(hours_remaining, 1)
        self.smoothing_factor = smoothing_factor

    def get_hourly_budget(self, spent_today: float) -> float:
        """
        Get recommended hourly budget.

        Args:
            spent_today: Amount spent today so far

        Returns:
            Recommended hourly budget
        """
        remaining_budget = max(0, self.daily_budget - spent_today)
        ideal_hourly = remaining_budget / self.hours_remaining

        # Apply smoothing factor
        return ideal_hourly * self.smoothing_factor

    def should_serve(
        self,
        candidate: AdCandidate,
        spent_this_hour: float,
        hourly_budget: float,
    ) -> bool:
        """
        Determine if ad should be served based on pacing.

        Args:
            candidate: Ad candidate
            spent_this_hour: Amount spent this hour
            hourly_budget: Budget for this hour

        Returns:
            True if ad should be served
        """
        if spent_this_hour >= hourly_budget:
            return False

        # Use probabilistic pacing
        remaining_ratio = (hourly_budget - spent_this_hour) / hourly_budget
        return remaining_ratio > 0.1  # Serve if >10% budget remaining

    def adjust_bid(
        self,
        bid: float,
        spent_today: float,
        target_spend: float,
    ) -> float:
        """
        Adjust bid based on pacing status.

        Args:
            bid: Original bid
            spent_today: Amount spent today
            target_spend: Target spend by this time

        Returns:
            Adjusted bid
        """
        if target_spend <= 0:
            return bid

        pacing_ratio = spent_today / target_spend

        if pacing_ratio < 0.8:
            # Under-pacing: increase bid
            return bid * 1.2
        elif pacing_ratio > 1.2:
            # Over-pacing: decrease bid
            return bid * 0.8
        else:
            # On track
            return bid
