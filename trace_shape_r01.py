"""Original finite temporal summaries for future controlled robustness probes.

These are features, not a trained predictor. No benchmark labels or problem IDs
are accepted. R00's upstream submitted baseline does not import this module.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def _slope(values: list[float]) -> float:
    """OLS against relative trace position; zero for a singleton."""
    if len(values) < 2:
        return 0.0
    xs = [i / (len(values) - 1) for i in range(len(values))]
    center = statistics.fmean(values)
    return sum((x - 0.5) * (y - center) for x, y in zip(xs, values)) / sum((x - 0.5)**2 for x in xs)


def _features(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f'{prefix}_{name}': 0.0 for name in (
            'front_tail_delta', 'position_slope', 'tail_std', 'excess_variation')}
    segment = max(1, len(values) // 3)
    front, tail = values[:segment], values[-segment:]
    variation = sum(abs(b - a) for a, b in zip(values, values[1:]))
    return {
        f'{prefix}_front_tail_delta': statistics.fmean(tail) - statistics.fmean(front),
        f'{prefix}_position_slope': _slope(values),
        f'{prefix}_tail_std': statistics.pstdev(tail),
        f'{prefix}_excess_variation': max(0.0, variation - abs(values[-1] - values[0])),
    }


def summarize_trace(entropy: Sequence[float], margin: Sequence[float],
                    chosen_logprob: Sequence[float], valid_mask: Sequence[bool]) -> dict[str, float | int]:
    """Mask EOS/padding before temporal segmentation; reject malformed traces.

    No constants calibrated on evaluation outcomes. Empty/singleton controls are
    finite but explicit via valid_token_count; they are not robustness labels.
    """
    count = len(entropy)
    if any(len(x) != count for x in (margin, chosen_logprob, valid_mask)):
        raise ValueError('aligned token arrays and mask required')
    if any(type(value) is not bool for value in valid_mask):
        raise ValueError('native boolean validity mask required')
    selected = []
    for values in (entropy, margin, chosen_logprob):
        row = []
        for value, valid in zip(values, valid_mask):
            if valid:
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError('finite numeric valid-token signals required')
                row.append(float(value))
        selected.append(row)
    result: dict[str, float | int] = {'valid_token_count': len(selected[0])}
    for values, name in zip(selected, ('entropy', 'margin', 'logprob')):
        result.update(_features(values, name))
    return result
