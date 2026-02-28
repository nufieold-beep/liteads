"""
Analytics & Reporting Router – Campaign performance, demand & supply reports.

Endpoints:
    GET  /campaigns                    – All campaigns summary
    GET  /campaign/{id}/realtime       – Current-hour live stats
    GET  /campaign/{id}/today          – Today aggregate stats
    GET  /campaign/{id}/budget         – Budget & spend status
    GET  /campaign/{id}/historical     – DB-based historical data
    GET  /reports/demand               – Demand report (adomain, gross rev, eCPM …)
    GET  /reports/supply               – Supply / publisher report
    POST /flush                        – Flush Redis stats → DB
    POST /sync-spend                   – Sync spend Redis → Campaign DB
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from liteads.ad_server.services.analytics_service import AnalyticsService
from liteads.common.database import get_session
from liteads.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DemandReportRow(BaseModel):
    """Single row in the demand report."""

    adomain: str = Field(..., description="Advertiser domain")
    demand_id: int | None = Field(None, description="Campaign / demand source ID")
    demand_creative_id: int | None = Field(None, description="Creative ID")
    impressions: int = 0
    gross_revenue: float = Field(0.0, description="Gross revenue in USD")
    bid_request_fill_rate: float = Field(0.0, description="Fill rate vs bid requests (%)")
    gross_ecpm: float = Field(0.0, description="Gross eCPM = (revenue / imps) * 1000")
    avg_win_price: float = Field(0.0, description="Average auction clearing price")
    bid_request_ecpm: float = Field(0.0, description="eCPM relative to bid requests")


class SupplyReportRow(BaseModel):
    """Single row in the supply / publisher report."""

    source_name: str = Field(..., description="Supply source / SSP name")
    campaign_id: int = Field(..., description="Campaign ID")
    campaign_name: str = Field("", description="Campaign display name")
    country_code: str = Field("XX", description="ISO 3166-1 alpha-2")
    country: str = Field("", description="Country name")
    bundle_id: str = Field("unknown", description="App bundle ID")
    ad_requests: int = Field(0, description="Bid requests received")
    ad_opportunities: int = Field(0, description="Eligible impressions (bid opps)")
    impressions: int = 0
    channel_revenue: float = Field(0.0, description="Revenue at auction clearing price")
    channel_ecpm: float = Field(0.0, description="Channel eCPM (clearing-based)")
    total_revenue: float = Field(0.0, description="Gross revenue (CPM cost-based)")
    ecpm: float = Field(0.0, description="eCPM (CPM cost / imps * 1000)")
    fill_rate_ad_req: float = Field(0.0, description="Fill rate vs ad requests (%)")
    fill_rate_ad_ops: float = Field(0.0, description="Fill rate vs ad opportunities (%)")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def _get_analytics_service(
    session: AsyncSession = Depends(get_session),
) -> AnalyticsService:
    return AnalyticsService(session)


# ---------------------------------------------------------------------------
# Campaign-level endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/campaigns",
    summary="All campaigns summary",
    description="Returns a summary of all campaigns with today's Redis spend.",
)
async def list_campaigns_summary(
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    summaries = await service.get_all_campaigns_summary()
    return {"campaigns": summaries, "count": len(summaries)}


@router.get(
    "/campaign/{campaign_id}/realtime",
    summary="Real-time stats (current hour)",
    description=(
        "Live stats from Redis for the current (or specified) hour.  "
        "Includes ad_requests, ad_opportunities, wins, impressions, "
        "spend, gross_ecpm, avg_win_price, bid_request_ecpm, and fill rates."
    ),
)
async def campaign_realtime(
    campaign_id: int,
    hour: str | None = Query(None, description="Hour (YYYYMMDDHH), default=current"),
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    return await service.get_campaign_realtime_stats(campaign_id, hour)


@router.get(
    "/campaign/{campaign_id}/today",
    summary="Today's aggregate stats",
    description="Aggregates all hourly Redis buckets for today.",
)
async def campaign_today(
    campaign_id: int,
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    return await service.get_campaign_today_stats(campaign_id)


@router.get(
    "/campaign/{campaign_id}/budget",
    summary="Budget & spend status",
    description="Shows daily + total budget, spend, remaining, and pacing %.",
)
async def campaign_budget(
    campaign_id: int,
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    data = await service.get_campaign_budget_status(campaign_id)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data


@router.get(
    "/campaign/{campaign_id}/historical",
    summary="Historical hourly stats",
    description="Query HourlyStat DB table with optional date range.",
)
async def campaign_historical(
    campaign_id: int,
    start: str | None = Query(None, description="Start datetime ISO-8601"),
    end: str | None = Query(None, description="End datetime ISO-8601"),
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    start_dt = (
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    )
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None
    )
    rows = await service.get_campaign_historical_stats(campaign_id, start_dt, end_dt)
    return {"campaign_id": campaign_id, "hours": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Demand & Supply reports
# ---------------------------------------------------------------------------

@router.get(
    "/reports/demand",
    response_model=list[DemandReportRow],
    summary="Demand report",
    description=(
        "Demand-side report grouped by ADOMAIN × DEMAND_ID × DEMAND_CREATIVE_ID. "
        "Returns GROSS_REVENUE, BID_REQUEST_FILL_RATE, GROSS_ECPM, "
        "AVG_WIN_PRICE, and BID_REQUEST_ECPM."
    ),
)
async def demand_report(
    start: str | None = Query(None, description="Start datetime ISO-8601"),
    end: str | None = Query(None, description="End datetime ISO-8601"),
    campaign_id: int | None = Query(None, description="Filter by campaign"),
    service: AnalyticsService = Depends(_get_analytics_service),
) -> list[dict[str, Any]]:
    start_dt = (
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    )
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None
    )
    return await service.get_demand_report(start_dt, end_dt, campaign_id)


@router.get(
    "/reports/supply",
    response_model=list[SupplyReportRow],
    summary="Supply / publisher report",
    description=(
        "Supply-side report grouped by Source Name × Campaign × Country × Bundle. "
        "Returns Ad Requests, Ad Opportunities, Impressions, "
        "Channel Revenue, Channel eCPM, Total Revenue, eCPM, "
        "Fill Rate (Ad Req), Fill Rate (Ad Ops)."
    ),
)
async def supply_report(
    start: str | None = Query(None, description="Start datetime ISO-8601"),
    end: str | None = Query(None, description="End datetime ISO-8601"),
    campaign_id: int | None = Query(None, description="Filter by campaign"),
    service: AnalyticsService = Depends(_get_analytics_service),
) -> list[dict[str, Any]]:
    start_dt = (
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    )
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None
    )
    return await service.get_supply_report(start_dt, end_dt, campaign_id)


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/flush",
    summary="Flush Redis hourly stats → DB",
    description=(
        "Persists Redis hourly stat counters into the HourlyStat table. "
        "Should be called once per hour by a cron job or scheduler."
    ),
)
async def flush_stats(
    hour: str | None = Query(None, description="Hour (YYYYMMDDHH), default=previous hour"),
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    flushed = await service.flush_hourly_stats(hour)
    return {"flushed_campaigns": flushed, "hour": hour or "previous"}


@router.post(
    "/sync-spend",
    summary="Sync Redis budget spend → Campaign DB",
    description=(
        "Updates Campaign.spent_today and Campaign.spent_total from Redis. "
        "Should run periodically to keep the DB in sync."
    ),
)
async def sync_spend(
    service: AnalyticsService = Depends(_get_analytics_service),
) -> dict[str, Any]:
    updated = await service.sync_campaign_spend_to_db()
    return {"updated_campaigns": updated}
