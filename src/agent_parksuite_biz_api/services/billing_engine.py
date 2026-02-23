from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP


def _parse_hhmm(value: str) -> tuple[int, int]:
    hours, minutes = value.split(":", 1)
    return int(hours), int(minutes)


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
    weekdays = item.get("weekdays")
    if weekdays and ts.isoweekday() not in weekdays:
        return False

    window = item.get("time_window")
    if not window:
        return True

    start = window.get("start")
    end = window.get("end")
    if not (start and end):
        return True
    return _in_time_window(ts, str(start), str(end))


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


def simulate_fee(rule_payload: list[dict], entry_time: datetime, exit_time: datetime) -> dict:
    if exit_time <= entry_time:
        return {
            "duration_minutes": 0,
            "total_amount": Decimal("0.00"),
            "breakdown": [],
        }

    duration_minutes = int((exit_time - entry_time).total_seconds() // 60)

    segment_minutes: dict[int, int] = {}
    segment_day_minutes: dict[int, OrderedDict[str, int]] = {}
    cursor = entry_time
    while cursor < exit_time:
        matched_index = None
        for idx, item in enumerate(rule_payload):
            if _item_matches(cursor, item):
                matched_index = idx
                break

        if matched_index is not None:
            segment_minutes[matched_index] = segment_minutes.get(matched_index, 0) + 1
            day_key = cursor.date().isoformat()
            day_map = segment_day_minutes.setdefault(matched_index, OrderedDict())
            day_map[day_key] = day_map.get(day_key, 0) + 1

        cursor += timedelta(minutes=1)

    breakdown: list[dict] = []
    total_amount = Decimal("0.00")

    for segment_index, minutes in sorted(segment_minutes.items()):
        item = rule_payload[segment_index]
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
