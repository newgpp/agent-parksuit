from datetime import datetime
from decimal import Decimal

from agent_parksuite_biz_api.services.billing_engine import simulate_fee


def test_periodic_with_free_minutes_and_cap() -> None:
    payload = [
        {
            "name": "day_periodic",
            "type": "periodic",
            "time_window": {"start": "08:00", "end": "22:00"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 30,
            "max_charge": 6,
        }
    ]
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 1, 11, 0, 0))
    assert result["total_amount"] == Decimal("6.00")
    assert result["breakdown"][0]["capped"] is True


def test_free_night_segment() -> None:
    payload = [
        {
            "name": "night_free",
            "type": "free",
            "time_window": {"start": "22:00", "end": "08:00"},
        }
    ]
    result = simulate_fee(payload, datetime(2026, 2, 1, 23, 0, 0), datetime(2026, 2, 2, 1, 0, 0))
    assert result["total_amount"] == Decimal("0.00")
    assert result["breakdown"][0]["minutes"] == 120


def test_tiered_billing() -> None:
    payload = [
        {
            "name": "day_tiered",
            "type": "tiered",
            "time_window": {"start": "08:00", "end": "22:00"},
            "unit_minutes": 30,
            "tiers": [
                {"start_minute": 0, "end_minute": 60, "unit_price": 2},
                {"start_minute": 60, "end_minute": None, "unit_price": 3},
            ],
        }
    ]
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 1, 11, 0, 0))
    assert result["total_amount"] == Decimal("10.00")


def test_periodic_round_up_when_not_divisible_by_unit() -> None:
    payload = [
        {
            "name": "day_periodic_non_divisible",
            "type": "periodic",
            "time_window": {"start": "08:00", "end": "22:00"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 0,
        }
    ]
    # 65 minutes should be charged as 3 units (ceil(65/30)).
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 1, 10, 5, 0))
    assert result["duration_minutes"] == 65
    assert result["total_amount"] == Decimal("6.00")


def test_periodic_billing_across_days() -> None:
    payload = [
        {
            "name": "all_day_periodic",
            "type": "periodic",
            "time_window": {"start": "08:00", "end": "20:00"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 0,
            "max_charge": 20,
        }
    ]
    # Day1 09:00 -> Day3 15:10 within 08:00-20:00:
    # Day1: 660min -> 22 units -> 44 -> cap20
    # Day2: 720min -> 24 units -> 48 -> cap20
    # Day3: 430min -> 15 units -> 30 -> cap20
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 3, 15, 10, 0))
    assert result["duration_minutes"] == 3250
    assert result["breakdown"][0]["minutes"] == 1810
    assert result["breakdown"][0]["capped"] is True
    assert result["total_amount"] == Decimal("60.00")


def test_periodic_billing_across_days_with_night_periodic_cap() -> None:
    payload = [
        {
            "name": "day_periodic",
            "type": "periodic",
            "time_window": {"start": "08:00", "end": "20:00"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 0,
            "max_charge": 20,
        },
        {
            "name": "night_periodic",
            "type": "periodic",
            "time_window": {"start": "20:00", "end": "08:00"},
            "unit_minutes": 60,
            "unit_price": 2,
            "free_minutes": 0,
            "max_charge": 10,
        },
    ]
    # Day segment total: 20 + 20 + 20 = 60
    # Night segment total: (240min->8) + (720min->cap10) + (480min->cap10) = 28
    # Grand total: 88
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 3, 15, 10, 0))
    assert result["duration_minutes"] == 3250
    assert result["total_amount"] == Decimal("88.00")
    assert result["breakdown"][0]["segment_name"] == "day_periodic"
    assert result["breakdown"][0]["minutes"] == 1810
    assert result["breakdown"][0]["amount"] == Decimal("60.00")
    assert result["breakdown"][0]["capped"] is True
    assert result["breakdown"][1]["segment_name"] == "night_periodic"
    assert result["breakdown"][1]["minutes"] == 1440
    assert result["breakdown"][1]["amount"] == Decimal("28.00")
    assert result["breakdown"][1]["capped"] is True


def test_tiered_billing_across_days_with_night_free() -> None:
    payload = [
        {
            "name": "day_tiered",
            "type": "tiered",
            "time_window": {"start": "08:00", "end": "20:00"},
            "unit_minutes": 30,
            "free_minutes": 30,
            "tiers": [
                {"start_minute": 0, "end_minute": 120, "unit_price": 2},
                {"start_minute": 120, "end_minute": None, "unit_price": 3},
            ],
            "max_charge": 20,
        },
        {
            "name": "night_free",
            "type": "free",
            "time_window": {"start": "20:00", "end": "08:00"},
        },
    ]
    # Day1 (09:00-20:00): 660min -> after free30 => 21 units -> cap20
    # Day2 (08:00-20:00): 720min -> 24 units -> cap20
    # Day3 (08:00-08:29): 29min -> 1 unit -> 2
    # Night windows are all free. Total = 42
    result = simulate_fee(payload, datetime(2026, 2, 1, 9, 0, 0), datetime(2026, 2, 3, 8, 29, 0))
    assert result["duration_minutes"] == 2849
    assert result["total_amount"] == Decimal("42.00")
    assert result["breakdown"][0]["segment_name"] == "day_tiered"
    assert result["breakdown"][0]["minutes"] == 1409
    assert result["breakdown"][0]["amount"] == Decimal("42.00")
    assert result["breakdown"][0]["capped"] is True
    assert result["breakdown"][1]["segment_name"] == "night_free"
    assert result["breakdown"][1]["minutes"] == 1440
    assert result["breakdown"][1]["amount"] == Decimal("0.00")
