"""Shared numeric conversions, markers and loss summaries for the diagnostic report."""
import math
from typing import Any, List, Optional


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return list(values)
    result: List[Optional[float]] = []
    queue: List[float] = []
    running_sum = 0.0
    for value in values:
        if value is None:
            result.append(None)
            continue
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def mean_tail(values: List[Optional[float]], ratio: float, take_tail: bool) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    count = max(1, int(len(valid) * ratio))
    target = valid[-count:] if take_tail else valid[:count]
    return float(sum(target) / len(target))


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value
