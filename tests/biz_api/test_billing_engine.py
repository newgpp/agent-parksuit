from datetime import datetime
from decimal import Decimal

from agent_parksuite_biz_api.services.billing_engine import (
    _collect_segment_minutes_by_scan,
    collect_segment_minutes_by_window,
    simulate_fee,
)


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


def test_periodic_with_timezone_aware_input_should_use_default_window_timezone() -> None:
    payload = [
        {
            "name": "day_periodic",
            "type": "periodic",
            "time_window": {"start": "08:00", "end": "20:00"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 0,
        }
    ]
    # 01:00-02:00 UTC equals 09:00-10:00 in Asia/Shanghai.
    result = simulate_fee(
        payload,
        datetime.fromisoformat("2026-02-01T01:00:00+00:00"),
        datetime.fromisoformat("2026-02-01T02:00:00+00:00"),
    )
    assert result["duration_minutes"] == 60
    assert result["total_amount"] == Decimal("4.00")


def test_periodic_with_time_window_timezone_should_match_by_window_timezone() -> None:
    payload = [
        {
            "name": "utc_periodic",
            "type": "periodic",
            "time_window": {"start": "01:00", "end": "03:00", "timezone": "UTC"},
            "unit_minutes": 30,
            "unit_price": 2,
            "free_minutes": 0,
        }
    ]
    # 01:00-02:00 UTC is inside the UTC-configured window.
    result = simulate_fee(
        payload,
        datetime.fromisoformat("2026-02-01T01:00:00+00:00"),
        datetime.fromisoformat("2026-02-01T02:00:00+00:00"),
    )
    assert result["duration_minutes"] == 60
    assert result["total_amount"] == Decimal("4.00")


def test_collect_segment_minutes_by_window_matches_scan() -> None:
    cases = [
        (
            [
                {
                    "name": "day_periodic",
                    "type": "periodic",
                    "time_window": {"start": "08:00", "end": "20:00"},
                    "unit_minutes": 30,
                    "unit_price": 2,
                },
                {
                    "name": "night_free",
                    "type": "free",
                    "time_window": {"start": "20:00", "end": "08:00"},
                },
            ],
            datetime(2026, 2, 1, 9, 0, 0),
            datetime(2026, 2, 3, 8, 29, 0),
        ),
        (
            [
                {
                    "name": "utc_day_periodic",
                    "type": "periodic",
                    "time_window": {"start": "01:00", "end": "03:00", "timezone": "UTC"},
                    "unit_minutes": 30,
                    "unit_price": 2,
                },
                {
                    "name": "sh_night_free",
                    "type": "free",
                    "time_window": {"start": "22:00", "end": "08:00", "timezone": "Asia/Shanghai"},
                },
            ],
            datetime.fromisoformat("2026-02-01T00:30:00+00:00"),
            datetime.fromisoformat("2026-02-02T03:30:00+00:00"),
        ),
    ]

    for payload, entry_time, exit_time in cases:
        old_minutes, old_day_minutes = _collect_segment_minutes_by_scan(payload, entry_time, exit_time)
        new_minutes, new_day_minutes = collect_segment_minutes_by_window(payload, entry_time, exit_time)
        assert new_minutes == old_minutes
        assert {idx: dict(day_map) for idx, day_map in new_day_minutes.items()} == {
            idx: dict(day_map) for idx, day_map in old_day_minutes.items()
        }
