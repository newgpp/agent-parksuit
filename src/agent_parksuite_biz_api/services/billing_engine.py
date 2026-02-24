from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import math
from typing import Any
from zoneinfo import ZoneInfo


def _parse_hhmm(value: str) -> tuple[int, int]:
    hours, minutes = value.split(":", 1)
    return int(hours), int(minutes)


@lru_cache(maxsize=32)
def _load_timezone(tz_name: str) -> ZoneInfo:
    return ZoneInfo(tz_name)


def _to_named_timezone(ts: datetime, tz_name: str) -> datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(_load_timezone(tz_name))


def _resolve_window_timezone(item: dict) -> str:
    window = item.get("time_window") or {}
    return str(window.get("timezone", "Asia/Shanghai"))


def _to_base_timezone(ts: datetime, base_tz: ZoneInfo | None) -> datetime:
    if ts.tzinfo is None or base_tz is None:
        return ts
    return ts.astimezone(base_tz)


def _in_time_window(ts: datetime, start: str, end: str) -> bool:
    sh, sm = _parse_hhmm(start)
    eh, em = _parse_hhmm(end)

    current = ts.hour * 60 + ts.minute
    start_minute = sh * 60 + sm
    end_minute = eh * 60 + em

    if start_minute == end_minute:
        return True
    if start_minute < end_minute:
        return start_minute <= current < end_minute
    return current >= start_minute or current < end_minute


def _item_matches(ts: datetime, item: dict) -> bool:
    window = item.get("time_window") or {}
    window_timezone = _resolve_window_timezone(item)
    local_ts = _to_named_timezone(ts, window_timezone)

    weekdays = item.get("weekdays")
    if weekdays and local_ts.isoweekday() not in weekdays:
        return False

    if not window:
        return True

    start = window.get("start")
    end = window.get("end")
    if not (start and end):
        return True
    return _in_time_window(local_ts, str(start), str(end))


def _compute_tiered_amount(units: int, unit_minutes: int, tiers: list[dict]) -> Decimal:
    amount = Decimal("0.00")
    for unit_index in range(units):
        start_minute = unit_index * unit_minutes
        unit_price = None
        for tier in tiers:
            tier_start = int(tier.get("start_minute", 0))
            tier_end = tier.get("end_minute")
            if start_minute >= tier_start and (tier_end is None or start_minute < int(tier_end)):
                unit_price = Decimal(str(tier.get("unit_price", 0)))
                break
        if unit_price is None:
            unit_price = Decimal("0.00")
        amount += unit_price
    return amount


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[datetime, datetime]] = [sorted_intervals[0]]
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def _subtract_intervals(
    intervals: list[tuple[datetime, datetime]],
    covered: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    if not covered:
        return intervals

    covered_merged = _merge_intervals(covered)
    result: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        cursor = start
        for cover_start, cover_end in covered_merged:
            if cover_end <= cursor:
                continue
            if cover_start >= end:
                break
            if cover_start > cursor:
                result.append((cursor, min(cover_start, end)))
            cursor = max(cursor, cover_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def _iter_minute_points(
    origin: datetime,
    overall_end: datetime,
    start: datetime,
    end: datetime,
):
    if start >= end:
        return
    step = timedelta(minutes=1)
    start_offset_minutes = (start - origin).total_seconds() / 60
    n = max(0, math.ceil(start_offset_minutes))
    point = origin + n * step
    while point < end and point < overall_end:
        yield point
        point += step


def _iter_local_dates(start_date, end_date):
    cursor = start_date
    while cursor <= end_date:
        yield cursor
        cursor += timedelta(days=1)


def _build_segment_candidate_intervals(
    item: dict[str, Any],
    entry_time: datetime,
    exit_time: datetime,
) -> list[tuple[datetime, datetime]]:
    timezone_name = _resolve_window_timezone(item)
    local_entry = _to_named_timezone(entry_time, timezone_name)
    local_exit = _to_named_timezone(exit_time, timezone_name)
    local_tz = local_entry.tzinfo
    base_tz = entry_time.tzinfo

    window = item.get("time_window") or {}
    start_raw = window.get("start")
    end_raw = window.get("end")
    start_minute = None
    end_minute = None
    if start_raw and end_raw:
        sh, sm = _parse_hhmm(str(start_raw))
        eh, em = _parse_hhmm(str(end_raw))
        start_minute = sh * 60 + sm
        end_minute = eh * 60 + em

    weekdays = item.get("weekdays")
    candidates: list[tuple[datetime, datetime]] = []

    for local_date in _iter_local_dates(local_entry.date(), local_exit.date()):
        if weekdays and local_date.isoweekday() not in weekdays:
            continue

        day_start_local = datetime(local_date.year, local_date.month, local_date.day, tzinfo=local_tz)
        next_day_start_local = day_start_local + timedelta(days=1)

        day_intervals_local: list[tuple[datetime, datetime]]
        if start_minute is None or end_minute is None or start_minute == end_minute:
            day_intervals_local = [(day_start_local, next_day_start_local)]
        elif start_minute < end_minute:
            segment_start = day_start_local + timedelta(minutes=start_minute)
            segment_end = day_start_local + timedelta(minutes=end_minute)
            day_intervals_local = [(segment_start, segment_end)]
        else:
            day_intervals_local = [
                (day_start_local, day_start_local + timedelta(minutes=end_minute)),
                (day_start_local + timedelta(minutes=start_minute), next_day_start_local),
            ]

        for local_start, local_end in day_intervals_local:
            clipped_start = max(local_start, local_entry)
            clipped_end = min(local_end, local_exit)
            if clipped_start >= clipped_end:
                continue
            base_start = _to_base_timezone(clipped_start, base_tz)
            base_end = _to_base_timezone(clipped_end, base_tz)
            candidates.append((base_start, base_end))

    return _merge_intervals(candidates)


def _collect_segment_minutes_by_scan(
    normalized_payload: list[dict[str, Any]],
    entry_time: datetime,
    exit_time: datetime,
) -> tuple[dict[int, int], dict[int, OrderedDict[str, int]]]:
    """Deprecated: minute-by-minute scan implementation kept only for parity testing."""
    segment_minutes: dict[int, int] = {}
    segment_day_minutes: dict[int, OrderedDict[str, int]] = {}
    cursor = entry_time
    while cursor < exit_time:
        matched_index = None
        for idx, item in enumerate(normalized_payload):
            if _item_matches(cursor, item):
                matched_index = idx
                break

        if matched_index is not None:
            segment_minutes[matched_index] = segment_minutes.get(matched_index, 0) + 1
            matched_item = normalized_payload[matched_index]
            day_key = _to_named_timezone(cursor, _resolve_window_timezone(matched_item)).date().isoformat()
            day_map = segment_day_minutes.setdefault(matched_index, OrderedDict())
            day_map[day_key] = day_map.get(day_key, 0) + 1

        cursor += timedelta(minutes=1)
    return segment_minutes, segment_day_minutes


def collect_segment_minutes_by_window(
    normalized_payload: list[dict[str, Any]],
    entry_time: datetime,
    exit_time: datetime,
) -> tuple[dict[int, int], dict[int, OrderedDict[str, int]]]:
    segment_minutes: dict[int, int] = {}
    segment_day_minutes: dict[int, OrderedDict[str, int]] = {}
    covered_intervals: list[tuple[datetime, datetime]] = []

    for idx, item in enumerate(normalized_payload):
        candidate_intervals = _build_segment_candidate_intervals(item, entry_time, exit_time)
        active_intervals = _subtract_intervals(candidate_intervals, covered_intervals)
        if not active_intervals:
            continue

        timezone_name = _resolve_window_timezone(item)
        day_map = segment_day_minutes.setdefault(idx, OrderedDict())
        for interval_start, interval_end in active_intervals:
            for point in _iter_minute_points(entry_time, exit_time, interval_start, interval_end):
                segment_minutes[idx] = segment_minutes.get(idx, 0) + 1
                day_key = _to_named_timezone(point, timezone_name).date().isoformat()
                day_map[day_key] = day_map.get(day_key, 0) + 1

        covered_intervals = _merge_intervals([*covered_intervals, *active_intervals])

    return segment_minutes, segment_day_minutes


def simulate_fee(
    rule_payload: list[Any],
    entry_time: datetime,
    exit_time: datetime,
) -> dict:
    if exit_time <= entry_time:
        return {
            "duration_minutes": 0,
            "total_amount": Decimal("0.00"),
            "breakdown": [],
        }

    duration_minutes = int((exit_time - entry_time).total_seconds() // 60)
    normalized_payload: list[dict[str, Any]] = [
        item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item
        for item in rule_payload
    ]
    segment_minutes, segment_day_minutes = collect_segment_minutes_by_window(
        normalized_payload,
        entry_time,
        exit_time,
    )

    breakdown: list[dict] = []
    total_amount = Decimal("0.00")

    for segment_index, minutes in sorted(segment_minutes.items()):
        item = normalized_payload[segment_index]
        segment_type = str(item.get("type", "free"))
        segment_name = str(item.get("name", f"segment_{segment_index + 1}"))

        if segment_type == "free":
            amount = Decimal("0.00")
            free_minutes = minutes
            capped = False
        else:
            free_minutes = int(item.get("free_minutes", 0))
            unit_minutes = max(1, int(item.get("unit_minutes", 30)))
            amount = Decimal("0.00")
            remaining_free = free_minutes
            capped = False
            max_charge = item.get("max_charge")
            cap_value = Decimal(str(max_charge)) if max_charge is not None else None

            day_map = segment_day_minutes.get(segment_index, OrderedDict())
            for day_minutes in day_map.values():
                day_chargeable_minutes = max(0, day_minutes - remaining_free)
                remaining_free = max(0, remaining_free - day_minutes)
                units = (day_chargeable_minutes + unit_minutes - 1) // unit_minutes

                if segment_type == "periodic":
                    unit_price = Decimal(str(item.get("unit_price", 0)))
                    day_amount = Decimal(units) * unit_price
                elif segment_type == "tiered":
                    tiers = item.get("tiers", [])
                    day_amount = _compute_tiered_amount(units, unit_minutes, tiers)
                else:
                    day_amount = Decimal("0.00")

                if cap_value is not None and day_amount >= cap_value:
                    day_amount = cap_value
                    capped = True

                amount += day_amount

        amount = _quantize(amount)
        total_amount += amount

        breakdown.append(
            {
                "segment_name": segment_name,
                "segment_type": segment_type,
                "minutes": minutes,
                "amount": amount,
                "free_minutes": free_minutes,
                "capped": capped,
            }
        )

    return {
        "duration_minutes": duration_minutes,
        "total_amount": _quantize(total_amount),
        "breakdown": breakdown,
    }
