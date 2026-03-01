#!/usr/bin/env python3
"""
Initialize test data in the database for end-to-end testing.

Creates advertisers, campaigns, creatives, and targeting rules.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def init_database():
    """Initialize database with test data."""
    import asyncpg

    print("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "liteads"),
        password=os.getenv("DB_PASSWORD", "liteads_password"),
        database=os.getenv("DB_NAME", "liteads"),
    )

    print("Creating tables...")

    # Create tables
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS advertisers (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            company VARCHAR(255),
            contact_email VARCHAR(255),
            balance DECIMAL(12,4) DEFAULT 0.0000 NOT NULL,
            daily_budget DECIMAL(12,4) DEFAULT 0.0000 NOT NULL,
            status INTEGER DEFAULT 1 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id SERIAL PRIMARY KEY,
            advertiser_id INTEGER NOT NULL REFERENCES advertisers(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            budget_daily DECIMAL(12,4),
            budget_total DECIMAL(12,4),
            spent_today DECIMAL(12,4) DEFAULT 0.0000 NOT NULL,
            spent_total DECIMAL(12,4) DEFAULT 0.0000 NOT NULL,
            bid_type INTEGER DEFAULT 1 NOT NULL,
            bid_amount DECIMAL(12,4) DEFAULT 1.0000 NOT NULL,
            environment INTEGER,
            bid_floor DECIMAL(10,4) DEFAULT 0.0000 NOT NULL,
            floor_config JSON,
            adomain VARCHAR(255),
            iab_categories JSON,
            start_time TIMESTAMP WITH TIME ZONE,
            end_time TIMESTAMP WITH TIME ZONE,
            freq_cap_daily INTEGER DEFAULT 10 NOT NULL,
            freq_cap_hourly INTEGER DEFAULT 3 NOT NULL,
            status INTEGER DEFAULT 1 NOT NULL,
            impressions INTEGER DEFAULT 0 NOT NULL,
            completions INTEGER DEFAULT 0 NOT NULL,
            clicks INTEGER DEFAULT 0 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS creatives (
            id SERIAL PRIMARY KEY,
            campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            video_url VARCHAR(1024) NOT NULL,
            vast_url VARCHAR(1024),
            companion_image_url VARCHAR(1024),
            landing_url VARCHAR(1024) NOT NULL,
            creative_type INTEGER DEFAULT 2 NOT NULL,
            duration INTEGER DEFAULT 30 NOT NULL,
            width INTEGER DEFAULT 1920 NOT NULL,
            height INTEGER DEFAULT 1080 NOT NULL,
            bitrate INTEGER,
            mime_type VARCHAR(50) DEFAULT 'video/mp4' NOT NULL,
            skippable BOOLEAN DEFAULT TRUE NOT NULL,
            skip_after INTEGER DEFAULT 5 NOT NULL,
            placement INTEGER DEFAULT 1 NOT NULL,
            status INTEGER DEFAULT 1 NOT NULL,
            quality_score INTEGER DEFAULT 80 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS targeting_rules (
            id SERIAL PRIMARY KEY,
            campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            rule_type VARCHAR(50) NOT NULL,
            rule_value JSONB NOT NULL,
            is_include BOOLEAN DEFAULT TRUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ad_events (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(64) NOT NULL,
            campaign_id INTEGER,
            creative_id INTEGER,
            event_type INTEGER NOT NULL,
            event_time TIMESTAMP WITH TIME ZONE NOT NULL,
            user_id VARCHAR(64),
            ip_address VARCHAR(45),
            cost DECIMAL(10,6) DEFAULT 0.000000 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id SERIAL PRIMARY KEY,
            campaign_id INTEGER NOT NULL,
            stat_hour TIMESTAMP WITH TIME ZONE NOT NULL,
            impressions INTEGER DEFAULT 0 NOT NULL,
            clicks INTEGER DEFAULT 0 NOT NULL,
            conversions INTEGER DEFAULT 0 NOT NULL,
            cost DECIMAL(12,4) DEFAULT 0.0000 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    # Check if data exists
    existing_count = await conn.fetchval("SELECT COUNT(*) FROM advertisers")
    if existing_count > 0:
        print(f"Database already has {existing_count} advertisers, skipping insert...")
        await conn.close()
        return

    print("Inserting test data...")

    # Insert advertisers
    advertisers = [
        ("游戏广告商", "游戏公司A", "game@example.com", 50000.00),
        ("电商广告商", "电商公司B", "shop@example.com", 100000.00),
        ("金融广告商", "金融公司C", "finance@example.com", 80000.00),
        ("教育广告商", "教育公司D", "edu@example.com", 30000.00),
        ("本地生活", "生活服务E", "local@example.com", 20000.00),
    ]

    advertiser_ids = []
    for name, company, email, balance in advertisers:
        row = await conn.fetchrow(
            """
            INSERT INTO advertisers (name, company, contact_email, balance, status)
            VALUES ($1, $2, $3, $4, 1)
            RETURNING id
            """,
            name, company, email, balance
        )
        advertiser_ids.append(row['id'])

    print(f"  Inserted {len(advertisers)} advertisers (IDs: {advertiser_ids})")

    # Insert campaigns
    # Use dynamic advertiser IDs
    a1, a2, a3, a4, a5 = advertiser_ids
    campaigns = [
        # advertiser_id, name, budget_daily, budget_total, bid_type, bid_amount
        (a1, "CTV Pre-Roll Campaign", 1000.00, 30000.00, 1, 15.00),   # CPM
        (a1, "InApp Video Campaign", 500.00, 15000.00, 1, 8.00),      # CPM
        (a2, "CTV Mid-Roll Campaign", 5000.00, 100000.00, 1, 18.00),  # CPM
        (a2, "InApp Rewarded Video", 3000.00, 50000.00, 1, 6.00),     # CPM
        (a3, "CTV Post-Roll Campaign", 2000.00, 60000.00, 1, 10.00),  # CPM
        (a3, "InApp Interstitial", 1500.00, 45000.00, 1, 7.50),       # CPM
        (a4, "CTV Sports Campaign", 800.00, 24000.00, 1, 12.00),      # CPM
        (a5, "InApp Gaming Campaign", 2000.00, 40000.00, 1, 5.00),    # CPM
        (a5, "CTV News Campaign", 1000.00, 30000.00, 1, 9.00),        # CPM
        (a1, "InApp Lifestyle Campaign", 300.00, 9000.00, 1, 6.50),   # CPM
    ]

    campaign_ids = []
    for adv_id, name, budget_d, budget_t, bid_type, bid_amt in campaigns:
        row = await conn.fetchrow(
            """
            INSERT INTO campaigns (advertiser_id, name, budget_daily, budget_total,
                                   bid_type, bid_amount, status)
            VALUES ($1, $2, $3, $4, $5, $6, 1)
            RETURNING id
            """,
            adv_id, name, budget_d, budget_t, bid_type, bid_amt
        )
        campaign_ids.append(row['id'])

    print(f"  Inserted {len(campaigns)} campaigns (IDs: {campaign_ids})")

    # Insert creatives - use campaign IDs (video-only, matching ORM model)
    c = campaign_ids  # shorthand
    creatives = [
        (c[0], "CTV Pre-Roll 30s", "30-second CTV pre-roll video", 1, 1920, 1080),
        (c[0], "CTV Pre-Roll 15s", "15-second CTV pre-roll video", 1, 1920, 1080),
        (c[1], "InApp Video 30s", "30-second in-app video", 2, 1280, 720),
        (c[2], "CTV Mid-Roll 30s", "30-second CTV mid-roll video", 1, 1920, 1080),
        (c[2], "CTV Mid-Roll 15s", "15-second CTV mid-roll video", 1, 1920, 1080),
        (c[3], "InApp Rewarded 30s", "30-second rewarded video", 2, 1280, 720),
        (c[4], "CTV Post-Roll 30s", "30-second CTV post-roll video", 1, 1920, 1080),
        (c[5], "InApp Interstitial 15s", "15-second interstitial video", 2, 1280, 720),
        (c[6], "CTV Sports 30s", "30-second sports CTV video", 1, 1920, 1080),
        (c[7], "InApp Gaming 30s", "30-second gaming in-app video", 2, 1280, 720),
        (c[8], "CTV News 15s", "15-second news CTV video", 1, 1920, 1080),
        (c[9], "InApp Lifestyle 30s", "30-second lifestyle in-app video", 2, 1280, 720),
    ]

    for camp_id, title, desc, creative_type, width, height in creatives:
        await conn.execute(
            """
            INSERT INTO creatives (campaign_id, title, description, video_url, landing_url,
                                   creative_type, width, height, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
            """,
            camp_id, title, desc,
            f"https://cdn.example.com/video/{camp_id}_{width}x{height}.mp4",
            f"https://example.com/landing/{camp_id}",
            creative_type, width, height
        )

    print(f"  Inserted {len(creatives)} creatives")

    # Insert targeting rules - use campaign IDs
    # Note: rule_value must be a dict with specific keys based on rule_type:
    # - platform/os: {"values": [...]}
    # - geo: {"countries": [...], "cities": [...]}
    # - age: {"min": N, "max": N}
    import json
    targeting_rules = [
        (c[0], "os", json.dumps({"values": ["android", "ios"]}), True),
        (c[0], "geo", json.dumps({"countries": ["CN"]}), True),
        (c[1], "age", json.dumps({"min": 18, "max": 34}), True),
        (c[2], "os", json.dumps({"values": ["android"]}), True),
        (c[2], "geo", json.dumps({"countries": ["CN"]}), True),
        (c[4], "age", json.dumps({"min": 25, "max": 99}), True),
        (c[6], "interest", json.dumps({"values": ["programming", "technology"]}), True),
        (c[7], "geo", json.dumps({"countries": ["CN"]}), True),
    ]

    for camp_id, rule_type, rule_value, is_include in targeting_rules:
        await conn.execute(
            """
            INSERT INTO targeting_rules (campaign_id, rule_type, rule_value, is_include)
            VALUES ($1, $2, $3, $4)
            """,
            camp_id, rule_type, rule_value, is_include
        )

    print(f"  Inserted {len(targeting_rules)} targeting rules")

    await conn.close()
    print("\nDatabase initialization complete!")


async def verify_data():
    """Verify data was inserted correctly."""
    import asyncpg

    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "liteads"),
        password=os.getenv("DB_PASSWORD", "liteads_password"),
        database=os.getenv("DB_NAME", "liteads"),
    )

    print("\n" + "=" * 50)
    print("Database Summary:")
    print("=" * 50)

    advertisers = await conn.fetchval("SELECT COUNT(*) FROM advertisers")
    campaigns = await conn.fetchval("SELECT COUNT(*) FROM campaigns")
    creatives = await conn.fetchval("SELECT COUNT(*) FROM creatives")
    targeting = await conn.fetchval("SELECT COUNT(*) FROM targeting_rules")

    print(f"  Advertisers: {advertisers}")
    print(f"  Campaigns: {campaigns}")
    print(f"  Creatives: {creatives}")
    print(f"  Targeting Rules: {targeting}")

    print("\nActive Campaigns:")
    rows = await conn.fetch("""
        SELECT c.id, c.name, a.name as advertiser, c.bid_type, c.bid_amount, c.budget_daily
        FROM campaigns c
        JOIN advertisers a ON c.advertiser_id = a.id
        WHERE c.status = 1
        ORDER BY c.id
        LIMIT 10
    """)

    for row in rows:
        bid_type = "CPM" if row["bid_type"] == 1 else "CPC"
        print(f"  [{row['id']}] {row['name']} ({row['advertiser']}) - {bid_type} ${row['bid_amount']}")

    await conn.close()


def main():
    print("=" * 50)
    print("LiteAds Database Initialization")
    print("=" * 50)

    asyncio.run(init_database())
    asyncio.run(verify_data())


if __name__ == "__main__":
    main()
