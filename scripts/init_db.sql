-- =========================================================================
-- LiteAds Database Initialization – CPM CTV & In-App Video Only
-- =========================================================================
-- This script is run automatically when the PostgreSQL container starts.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -------------------------------------------------------------------------
-- Advertisers
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS advertisers (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(255)  NOT NULL,
    company         VARCHAR(255),
    contact_email   VARCHAR(255),
    contact_phone   VARCHAR(50),
    balance         DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    credit_limit    DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    status          SMALLINT      DEFAULT 1    NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_advertiser_status ON advertisers(status);

-- -------------------------------------------------------------------------
-- Campaigns  (CPM-only, environment = ctv | inapp)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS campaigns (
    id              BIGSERIAL PRIMARY KEY,
    advertiser_id   BIGINT        NOT NULL REFERENCES advertisers(id) ON DELETE CASCADE,
    name            VARCHAR(255)  NOT NULL,
    description     TEXT,
    environment     VARCHAR(10)   NOT NULL DEFAULT 'ctv',   -- 'ctv' | 'inapp'
    budget_daily    DECIMAL(15,2),
    budget_total    DECIMAL(15,2),
    spent_today     DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    spent_total     DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    bid_type        SMALLINT      DEFAULT 1    NOT NULL,    -- 1 = CPM (only)
    bid_amount      DECIMAL(10,4) DEFAULT 5.0000 NOT NULL,  -- CPM price
    start_time      TIMESTAMP WITH TIME ZONE,
    end_time        TIMESTAMP WITH TIME ZONE,
    freq_cap_daily  SMALLINT,
    freq_cap_hourly SMALLINT,
    status          SMALLINT      DEFAULT 1    NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_campaign_advertiser  ON campaigns(advertiser_id);
CREATE INDEX IF NOT EXISTS idx_campaign_status      ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaign_schedule    ON campaigns(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_campaign_environment ON campaigns(environment);

-- -------------------------------------------------------------------------
-- Creatives  (Video-only: CTV_VIDEO = 1, INAPP_VIDEO = 2)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS creatives (
    id                  BIGSERIAL PRIMARY KEY,
    campaign_id         BIGINT        NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    title               VARCHAR(255),
    description         TEXT,
    -- Video-specific fields
    video_url           VARCHAR(500)  NOT NULL,
    vast_url            VARCHAR(500),              -- External VAST tag (wrapper)
    companion_image_url VARCHAR(500),              -- Companion banner
    landing_url         VARCHAR(500)  NOT NULL,
    creative_type       SMALLINT      DEFAULT 1    NOT NULL,  -- 1=CTV_VIDEO, 2=INAPP_VIDEO
    duration            SMALLINT      DEFAULT 30   NOT NULL,  -- seconds
    bitrate             INTEGER       DEFAULT 2500,            -- kbps
    mime_type           VARCHAR(50)   DEFAULT 'video/mp4',
    width               SMALLINT      DEFAULT 1920,
    height              SMALLINT      DEFAULT 1080,
    skippable           BOOLEAN       DEFAULT FALSE,
    skip_after          SMALLINT      DEFAULT 5,               -- seconds
    placement           VARCHAR(20)   DEFAULT 'pre_roll',      -- pre_roll | mid_roll | post_roll
    quality_score       DECIMAL(5,4)  DEFAULT 0.5000,
    status              SMALLINT      DEFAULT 1    NOT NULL,
    impressions         BIGINT        DEFAULT 0    NOT NULL,
    clicks              BIGINT        DEFAULT 0    NOT NULL,
    video_starts        BIGINT        DEFAULT 0    NOT NULL,
    video_completions   BIGINT        DEFAULT 0    NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_creative_campaign ON creatives(campaign_id);
CREATE INDEX IF NOT EXISTS idx_creative_status   ON creatives(status);
CREATE INDEX IF NOT EXISTS idx_creative_type     ON creatives(creative_type);

-- -------------------------------------------------------------------------
-- Targeting Rules
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS targeting_rules (
    id          BIGSERIAL PRIMARY KEY,
    campaign_id BIGINT      NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    rule_type   VARCHAR(50) NOT NULL,     -- environment, device, os, geo, app_bundle, content_genre, daypart
    rule_value  JSONB       NOT NULL,
    is_include  BOOLEAN     DEFAULT TRUE NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_targeting_campaign ON targeting_rules(campaign_id);
CREATE INDEX IF NOT EXISTS idx_targeting_type     ON targeting_rules(rule_type);

-- -------------------------------------------------------------------------
-- Ad Events  (VAST video events + win/billing)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_events (
    id             BIGSERIAL PRIMARY KEY,
    request_id     VARCHAR(64)   NOT NULL,
    campaign_id    BIGINT        REFERENCES campaigns(id) ON DELETE SET NULL,
    creative_id    BIGINT        REFERENCES creatives(id) ON DELETE SET NULL,
    event_type     SMALLINT      NOT NULL,     -- See EventType enum (1=impression .. 14=error)
    event_time     TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id        VARCHAR(64),
    ip_address     VARCHAR(45),
    environment    VARCHAR(10)   DEFAULT 'ctv',
    video_position DECIMAL(5,2),               -- Playback position in seconds
    cost           DECIMAL(10,6) DEFAULT 0.000000 NOT NULL,
    auction_price  DECIMAL(10,6) DEFAULT 0.000000,  -- Clearing price from nurl/burl
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_request   ON ad_events(request_id);
CREATE INDEX IF NOT EXISTS idx_event_campaign  ON ad_events(campaign_id);
CREATE INDEX IF NOT EXISTS idx_event_time      ON ad_events(event_time);
CREATE INDEX IF NOT EXISTS idx_event_type_time ON ad_events(event_type, event_time);
CREATE INDEX IF NOT EXISTS idx_event_env       ON ad_events(environment);

-- -------------------------------------------------------------------------
-- Hourly Stats  (Video metrics for fill-rate / VTR optimisation)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hourly_stats (
    id                BIGSERIAL PRIMARY KEY,
    campaign_id       BIGINT        NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    creative_id       BIGINT        REFERENCES creatives(id) ON DELETE SET NULL,
    stat_hour         TIMESTAMP WITH TIME ZONE NOT NULL,
    impressions       BIGINT        DEFAULT 0 NOT NULL,
    clicks            BIGINT        DEFAULT 0 NOT NULL,
    video_starts      BIGINT        DEFAULT 0 NOT NULL,
    video_first_quartiles BIGINT    DEFAULT 0 NOT NULL,
    video_midpoints   BIGINT        DEFAULT 0 NOT NULL,
    video_third_quartiles BIGINT    DEFAULT 0 NOT NULL,
    video_completions BIGINT        DEFAULT 0 NOT NULL,
    video_skips       BIGINT        DEFAULT 0 NOT NULL,
    vtr               DECIMAL(5,4)  DEFAULT 0.0000,   -- completions / starts
    cost              DECIMAL(15,4) DEFAULT 0.0000 NOT NULL,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stats_campaign_hour ON hourly_stats(campaign_id, stat_hour);
CREATE INDEX IF NOT EXISTS idx_stats_hour          ON hourly_stats(stat_hour);

-- =========================================================================
-- Seed Data – CTV & In-App Video Campaigns
-- =========================================================================

INSERT INTO advertisers (name, company, balance, status) VALUES
    ('CTV Demo Advertiser',    'StreamCo Inc.',     50000.00, 1),
    ('InApp Video Advertiser', 'GameMedia Corp.',   30000.00, 1);

-- CTV campaign  (CPM $12)
INSERT INTO campaigns (advertiser_id, name, environment, budget_daily, budget_total, bid_type, bid_amount, status) VALUES
    (1, 'CTV Pre-Roll – Premium',   'ctv',   500.00, 10000.00, 1, 12.0000, 1),
    (1, 'CTV Mid-Roll – Sports',    'ctv',   300.00,  6000.00, 1,  8.0000, 1);

-- InApp video campaign  (CPM $6)
INSERT INTO campaigns (advertiser_id, name, environment, budget_daily, budget_total, bid_type, bid_amount, status) VALUES
    (2, 'InApp Video – Casual Games', 'inapp', 200.00,  4000.00, 1, 6.0000, 1),
    (2, 'InApp Video – Streaming',    'inapp', 250.00,  5000.00, 1, 7.5000, 1);

-- Creatives  (CTV)
INSERT INTO creatives (campaign_id, title, description, video_url, companion_image_url, landing_url, creative_type,
                       duration, bitrate, mime_type, width, height, skippable, skip_after, placement, status) VALUES
    (1, 'CTV 30s Pre-Roll', 'Premium CTV pre-roll ad',
     'https://cdn.example.com/video/ctv_preroll_30s.mp4', 'https://cdn.example.com/img/companion_300x250.png',
     'https://example.com/landing/ctv1', 1, 30, 5000, 'video/mp4', 1920, 1080, TRUE, 5, 'pre_roll', 1),
    (2, 'CTV 15s Mid-Roll', 'Sports mid-roll ad',
     'https://cdn.example.com/video/ctv_midroll_15s.mp4', NULL,
     'https://example.com/landing/ctv2', 1, 15, 4000, 'video/mp4', 1920, 1080, FALSE, 0, 'mid_roll', 1);

-- Creatives  (InApp)
INSERT INTO creatives (campaign_id, title, description, video_url, companion_image_url, landing_url, creative_type,
                       duration, bitrate, mime_type, width, height, skippable, skip_after, placement, status) VALUES
    (3, 'InApp 15s Rewarded', 'Rewarded video for casual games',
     'https://cdn.example.com/video/inapp_reward_15s.mp4', 'https://cdn.example.com/img/inapp_comp.png',
     'https://example.com/landing/inapp1', 2, 15, 2500, 'video/mp4', 1280, 720, FALSE, 0, 'pre_roll', 1),
    (4, 'InApp 30s Interstitial', 'Streaming app interstitial',
     'https://cdn.example.com/video/inapp_interstitial_30s.mp4', NULL,
     'https://example.com/landing/inapp2', 2, 30, 3000, 'video/mp4', 1280, 720, TRUE, 5, 'pre_roll', 1);

-- Targeting Rules
INSERT INTO targeting_rules (campaign_id, rule_type, rule_value, is_include) VALUES
    -- CTV campaign 1: target Roku + Fire TV in US
    (1, 'environment', '{"values": ["ctv"]}', TRUE),
    (1, 'device',      '{"os": ["roku", "firetv", "tvos"]}', TRUE),
    (1, 'geo',         '{"countries": ["US"], "dma": ["501", "803"]}', TRUE),
    -- CTV campaign 2: target all CTV devices
    (2, 'environment', '{"values": ["ctv"]}', TRUE),
    (2, 'content_genre', '{"values": ["sports", "entertainment"]}', TRUE),
    -- InApp campaign 3: target mobile devices
    (3, 'environment', '{"values": ["inapp"]}', TRUE),
    (3, 'device',      '{"os": ["android", "ios"]}', TRUE),
    -- InApp campaign 4: target all InApp
    (4, 'environment', '{"values": ["inapp"]}', TRUE);

DO $$
BEGIN
    RAISE NOTICE 'LiteAds database initialized – CPM CTV & In-App Video Only!';
END $$;
