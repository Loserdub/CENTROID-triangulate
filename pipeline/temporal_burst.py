#!/usr/bin/env python3
"""
CENTROID Pipeline — Temporal Half-Life Decay & Kleinberg Burst Detection
Location: pipeline/temporal_burst.py

Description:
  Implements:
    1. Exponential Temporal Half-Life Decay: w(t) = exp(-lambda * delta_t) with lambda = ln(2) / half_life.
    2. Kleinberg 2-State Automaton Burst Detection: Pure deterministic dynamic programming
       to isolate authentic accelerating interest from single-day spam spikes.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_decay_lambda(half_life_days: float = 14.0) -> float:
    """Computes exponential decay constant lambda = ln(2) / half_life."""
    hl = max(0.001, float(half_life_days))
    return math.log(2.0) / hl


def apply_exponential_temporal_decay(
    values: List[float],
    timestamps: Optional[List[str]] = None,
    half_life_days: float = 14.0,
    reference_dt: Optional[datetime] = None
) -> List[float]:
    """
    Applies exponential temporal decay w(t) = exp(-lambda * delta_t) to a series.
    If timestamps are provided (ISO strings), delta_t is measured in days from reference_dt (or latest ts).
    If timestamps are omitted, delta_t is the epoch index distance from the latest epoch.
    """
    if not values:
        return []

    lam = compute_decay_lambda(half_life_days)
    n = len(values)

    if timestamps and len(timestamps) == n:
        parsed_dts = []
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                parsed_dts.append(dt)
            except Exception:
                parsed_dts.append(None)

        # Determine reference time
        ref = reference_dt
        if ref is None:
            valid_dts = [d for d in parsed_dts if d is not None]
            ref = max(valid_dts) if valid_dts else datetime.now(timezone.utc)

        weights = []
        for i, dt in enumerate(parsed_dts):
            if dt is not None:
                delta_days = max(0.0, (ref - dt).total_seconds() / 86400.0)
            else:
                delta_days = float(n - 1 - i)
            weights.append(math.exp(-lam * delta_days))
    else:
        # Uniform epoch spacing decay
        weights = [math.exp(-lam * float(n - 1 - i)) for i in range(n)]

    return [round(v * w, 4) for v, w in zip(values, weights)]


def detect_kleinberg_burst_2state(
    counts: List[int],
    total_counts: List[int],
    s: float = 2.0,
    gamma: float = 1.0
) -> Dict[str, Any]:
    """
    Jon Kleinberg's 2-State Automaton for Burst Detection.
    State 0 = Baseline activity rate p0
    State 1 = Burst activity rate p1 = s * p0

    State transition cost:
      0 -> 1: tau = gamma * ln(T + 1)  (penalty for entering burst)
      1 -> 0: 0.0
      i -> i: 0.0

    Cost function: Negative log-likelihood under binomial distribution.
    Determines optimal Viterbi state sequence, separating authentic sustained interest
    from single-day spam spikes.
    """
    T = len(counts)
    if T == 0 or len(total_counts) != T:
        return {
            "is_bursting": False,
            "current_state": 0,
            "state_sequence": [],
            "burst_weight": 0.0,
            "burst_intensity": 1.0,
            "status": "INSUFFICIENT_DATA"
        }

    sum_k = sum(counts)
    sum_n = sum(total_counts)

    if sum_n <= 0:
        return {
            "is_bursting": False,
            "current_state": 0,
            "state_sequence": [0] * T,
            "burst_weight": 0.0,
            "burst_intensity": 1.0,
            "status": "BASELINE"
        }

    # Baseline probability p0
    p0 = max(1e-6, min(0.999999, sum_k / sum_n))
    # Burst state probability p1 = s * p0
    s_mult = max(1.1, float(s))
    p1 = min(0.999999, max(p0 * 1.05, p0 * s_mult))

    # Transition penalty tau = gamma * ln(T + 1)
    tau = max(0.1, gamma * math.log(T + 1.0))

    # Precompute log probabilities
    ln_p0 = math.log(p0)
    ln_1_p0 = math.log(1.0 - p0)
    ln_p1 = math.log(p1)
    ln_1_p1 = math.log(1.0 - p1)

    def sigma(state: int, t_idx: int) -> float:
        k = counts[t_idx]
        n_t = max(k, total_counts[t_idx])
        if state == 0:
            return -(k * ln_p0 + (n_t - k) * ln_1_p0)
        else:
            return -(k * ln_p1 + (n_t - k) * ln_1_p1)

    # Dynamic programming matrices for Viterbi path
    # cost[state, t]
    c0 = [0.0] * T
    c1 = [0.0] * T
    bp0 = [0] * T
    bp1 = [0] * T

    # Initialization t = 0
    c0[0] = sigma(0, 0)
    c1[0] = sigma(1, 0) + tau  # must pay penalty to begin in burst state

    for t in range(1, T):
        # State 0 transitions from (0 -> 0) cost 0, (1 -> 0) cost 0
        from_0_to_0 = c0[t - 1]
        from_1_to_0 = c1[t - 1]
        if from_0_to_0 <= from_1_to_0:
            c0[t] = sigma(0, t) + from_0_to_0
            bp0[t] = 0
        else:
            c0[t] = sigma(0, t) + from_1_to_0
            bp0[t] = 1

        # State 1 transitions from (0 -> 1) cost tau, (1 -> 1) cost 0
        from_0_to_1 = c0[t - 1] + tau
        from_1_to_1 = c1[t - 1]
        if from_0_to_1 <= from_1_to_1:
            c1[t] = sigma(1, t) + from_0_to_1
            bp1[t] = 0
        else:
            c1[t] = sigma(1, t) + from_1_to_1
            bp1[t] = 1

    # Backtrace
    states = [0] * T
    if c1[T - 1] < c0[T - 1]:
        states[T - 1] = 1
    else:
        states[T - 1] = 0

    for t in range(T - 1, 0, -1):
        if states[t] == 1:
            states[t - 1] = bp1[t]
        else:
            states[t - 1] = bp0[t]

    # Calculate burst weight: sum over burst states of (sigma_0 - sigma_1)
    total_burst_weight = 0.0
    burst_spans = 0
    for t in range(T):
        if states[t] == 1:
            diff = sigma(0, t) - sigma(1, t)
            if diff > 0:
                total_burst_weight += diff
            burst_spans += 1

    is_currently_bursting = (states[-1] == 1)

    # Status classification
    if is_currently_bursting and burst_spans >= 2:
        status = "AUTHENTIC BURST"
    elif is_currently_bursting and burst_spans == 1:
        # Check if single epoch count is anomalous or spam spike
        status = "EMERGING BURST"
    elif not is_currently_bursting and counts[-1] > p0 * total_counts[-1] * 2.0:
        # High count in latest epoch that failed the transition cost threshold
        status = "SPAM SPIKE DAMPENED"
    else:
        status = "BASELINE ACTIVITY"

    return {
        "is_bursting": is_currently_bursting,
        "current_state": states[-1],
        "state_sequence": states,
        "burst_weight": round(total_burst_weight, 3),
        "burst_intensity": round(p1 / max(1e-6, p0), 2),
        "burst_epochs_count": burst_spans,
        "status": status
    }


def evaluate_term_burst_kinematics(
    word: str,
    epoch_counts: List[int],
    epoch_totals: List[int],
    timestamps: Optional[List[str]] = None,
    half_life_days: float = 14.0
) -> Dict[str, Any]:
    """
    Integrates exponential temporal decay and Kleinberg burst detection
    to evaluate term momentum and growth authenticity.
    """
    decayed_counts = apply_exponential_temporal_decay(
        [float(c) for c in epoch_counts],
        timestamps=timestamps,
        half_life_days=half_life_days
    )
    burst_res = detect_kleinberg_burst_2state(epoch_counts, epoch_totals, s=2.5, gamma=1.0)

    # Net velocity
    if len(epoch_counts) >= 2:
        c_prev = epoch_counts[-2]
        c_curr = epoch_counts[-1]
        pct_growth = round(((c_curr - c_prev) / max(1, c_prev)) * 100.0, 1)
    else:
        pct_growth = 0.0

    return {
        "word": word,
        "latest_count": epoch_counts[-1] if epoch_counts else 0,
        "decayed_weight": decayed_counts[-1] if decayed_counts else 0.0,
        "growth_pct": pct_growth,
        "is_bursting": burst_res["is_bursting"],
        "burst_weight": burst_res["burst_weight"],
        "burst_intensity": burst_res["burst_intensity"],
        "status": burst_res["status"]
    }
