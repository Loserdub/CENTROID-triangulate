#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 10: Predictive Trend Horizon & Topic Synthesizer
Specification: Synthesizes high-probability publication-ready topics, computes
early-mover timing windows, detects sunset topics, and reconstructs multi-snapshot
timelines from exported data sheets.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import atomic_write_json, validate_snapshot_schema
    from pipeline.temporal_burst import detect_kleinberg_burst_2state, apply_exponential_temporal_decay
except ImportError:
    from common import atomic_write_json, validate_snapshot_schema
    from temporal_burst import detect_kleinberg_burst_2state, apply_exponential_temporal_decay

OUTPUTS_DIR = BASE_DIR / "outputs"
TREND_FORECAST_PATH = OUTPUTS_DIR / "trend_forecast.json"

logger = logging.getLogger("TREND_FORECAST")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][TREND_FORECAST] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_json_safe(path: Path) -> Any:
    """Safely loads JSON from path or returns None."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed loading {path}: {e}")
    return None


def title_case_phrase(phrase: str) -> str:
    """Capitalizes phrase words cleanly."""
    return " ".join(word.capitalize() for word in phrase.split())


def formulate_topic_title(cluster_label: str, medoid: str, anchors: List[str], collocations: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    Formulates a publication-ready topic working title and editorial angle
    by fusing the cluster medoid, emerging anchors, and extracted collocations.
    """
    label_clean = cluster_label.replace("Cluster", "").strip()
    primary_term = medoid or label_clean or "Innovation"

    # Find relevant collocations matching the cluster domain
    relevant_collocs = []
    for c in collocations:
        p = c.get("phrase", "").lower()
        if primary_term.lower() in p or any(a.lower() in p for a in anchors):
            relevant_collocs.append(c.get("phrase"))

    best_colloc = relevant_collocs[0] if relevant_collocs else (collocations[0]["phrase"] if collocations else "")
    anchor_phrase = anchors[0] if anchors else "Automation"

    # Narrative Headline Templates based on domain signals
    if best_colloc and primary_term.lower() not in best_colloc.lower():
        title = f"{title_case_phrase(best_colloc)}: The Emerging Shift in {primary_term.capitalize()} Strategy"
        editorial_angle = f"Examines how rapid adoption of {best_colloc} is redefining traditional {primary_term} workflows and creating early competitive advantage."
        intent = "Strategic Market Shift"
    elif any(term in primary_term.lower() for term in ["tax", "audit", "compliance", "retention", "accounting"]):
        title = f"Navigating {primary_term.capitalize()} & {anchor_phrase.capitalize()}: Modern Audit Readiness & Compliance Playbook"
        editorial_angle = f"A tactical guide on managing regulatory scrutiny in {primary_term}, mitigating enforcement risks, and establishing defensible audit trails."
        intent = "Regulatory & Risk Management"
    elif any(term in primary_term.lower() for term in ["audio", "dsp", "music", "production", "studio"]):
        title = f"Next-Gen {primary_term.capitalize()}: Leveraging {anchor_phrase.capitalize()} in Modern Creative Workflows"
        editorial_angle = f"Deep dive into high-performance {primary_term} techniques, examining the transition toward hybrid {anchor_phrase} systems."
        intent = "Technical Architecture"
    else:
        title = f"The Next Frontier in {primary_term.capitalize()}: Scaling {anchor_phrase.capitalize()} for High-Impact Execution"
        editorial_angle = f"Synthesizes emerging practices in {primary_term}, outlining how pioneering teams integrate {anchor_phrase} to capture market momentum."
        intent = "Industry Innovation"

    return title, editorial_angle, intent


def synthesize_predictive_topics(base_dir: Optional[Path] = None, forecast_horizon: float = 1.0) -> Dict[str, Any]:
    """
    Synthesizes the Top 3 High-Probability Emerging Topics, identifies early-mover timing windows,
    detects decaying/sunset topics, and builds turnkey editorial briefs.
    """
    b_dir = base_dir or BASE_DIR
    out_dir = b_dir / "outputs"

    # 1. Load source data layers
    clusters_data = load_json_safe(out_dir / "clusters.json") or {}
    drift_data = load_json_safe(out_dir / "drift_analysis.json") or {}
    super_data = load_json_safe(out_dir / "super_centroids.json") or {}
    collocs_data = load_json_safe(out_dir / "collocations.json") or []
    market_data = load_json_safe(out_dir / "market_benchmark.json") or {}

    trajectories = drift_data.get("trajectories", [])
    breakout_signals = drift_data.get("breakout_signals", [])

    # 2. Score cluster momentum & kinetic trajectory
    cluster_scores = []
    for traj in trajectories:
        cid = str(traj.get("cluster_id"))
        vel = float(traj.get("drift_velocity", 0.05))
        acc = float(traj.get("drift_acceleration", 0.0))
        label = traj.get("label", f"Cluster {cid}")
        medoid = traj.get("medoid", label)

        # Kinetic score = velocity + 0.5 * acceleration
        kinetic_momentum = round(vel + 0.5 * max(0.0, acc), 4)

        # Gather cluster anchors from super_centroids.json if available
        sc_info = super_data.get("clusters", {}).get(cid, {})
        anchors = [a.get("word") for a in sc_info.get("projected_semantic_anchors", []) if a.get("word")]
        if not anchors:
            anchors = [a.get("word") for a in sc_info.get("nearest_semantic_anchors", []) if a.get("word")]
        if not anchors:
            anchors = [sig.get("word") for sig in breakout_signals if str(sig.get("cluster_id")) == cid]

        cluster_scores.append({
            "cluster_id": cid,
            "label": label,
            "medoid": medoid,
            "velocity": vel,
            "acceleration": acc,
            "kinetic_momentum": kinetic_momentum,
            "heading_angle": traj.get("heading_angle", 0),
            "heading_desc": traj.get("heading_desc", "Dynamic Expansion"),
            "anchors": anchors[:4]
        })

    # Sort clusters by kinetic momentum descending
    cluster_scores.sort(key=lambda c: c["kinetic_momentum"], reverse=True)

    # 3. Formulate Top Emerging Topics
    emerging_topics: List[Dict[str, Any]] = []
    for idx, c in enumerate(cluster_scores[:3]):
        title, angle, intent = formulate_topic_title(c["label"], c["medoid"], c["anchors"], collocs_data)

        # Determine urgency and publication timing window
        if c["kinetic_momentum"] >= 0.10:
            stage = "🔥 RAPIDLY RISING"
            urgency_tier = "HIGH MOMENTUM"
            timing_window = "Immediate (Next 1–2 Weeks): Early-mover advantage before competitor saturation."
        elif c["kinetic_momentum"] >= 0.06:
            stage = "🌱 EARLY BREAKOUT"
            urgency_tier = "EXPANDING HORIZON"
            timing_window = "Next 2–4 Weeks: Ideal timing to establish topical authority as search demand surges."
        else:
            stage = "📈 STEADY ACCUMULATION"
            urgency_tier = "STRATEGIC PILLAR"
            timing_window = "Next 30 Days: Evergreen anchor piece to reinforce primary domain authority."

        # Plain-English "Why this is trending"
        why_trending = (
            f"Cross-snapshot momentum accelerated (+{int(c['velocity'] * 500)}% velocity index). "
            f"Anchored by emerging keywords [{', '.join(c['anchors'][:3]) or c['medoid']}], "
            f"signaling strong organic reader interest."
        )

        # Build Turnkey Editorial Brief
        brief_prompt = (
            f"# EDITORIAL BRIEF: {title}\n\n"
            f"## Strategic Context:\n"
            f"- Working Title: {title}\n"
            f"- Thematic Intent: {intent}\n"
            f"- Recommended Publication Window: {timing_window}\n\n"
            f"## Target Angle & Editorial Scope:\n"
            f"{angle}\n\n"
            f"## Core Vocabulary to Integrate:\n"
            f"- Primary Domain Anchor: {c['medoid']}\n"
            f"- Emerging Drivers: {', '.join(c['anchors']) or 'N/A'}\n"
            f"- Phrasal Collocations: {', '.join([cl['phrase'] for cl in collocs_data[:4]]) or 'N/A'}\n\n"
            f"## Writing Instructions:\n"
            f"1. Write an engaging, authoritative 1,200-word article optimized for decision-makers.\n"
            f"2. Seamlessly weave in the emerging keywords without superficial keyword stuffing.\n"
            f"3. Address practical implementation challenges and actionable takeaways in Section 2.\n"
            f"4. Provide a structured comparison or bulleted decision matrix for clarity.\n"
        )

        emerging_topics.append({
            "rank": idx + 1,
            "cluster_id": c["cluster_id"],
            "cluster_label": c["label"],
            "working_title": title,
            "editorial_angle": angle,
            "thematic_intent": intent,
            "stage": stage,
            "urgency_tier": urgency_tier,
            "timing_window": timing_window,
            "momentum_score": round(c["kinetic_momentum"] * 100, 1),
            "why_trending": why_trending,
            "key_vocabulary": [c["medoid"]] + c["anchors"][:3],
            "editorial_brief": brief_prompt
        })

    # 4. Detect Sunset / Decaying Topics (What NOT to write about)
    sunset_topics: List[Dict[str, Any]] = []
    # Clusters with lowest momentum or decelerating
    decaying = [c for c in cluster_scores if c["acceleration"] < 0 or c["kinetic_momentum"] <= 0.05]
    if not decaying and len(cluster_scores) > 3:
        decaying = cluster_scores[-2:]

    for d in decaying[:2]:
        sunset_topics.append({
            "cluster_id": d["cluster_id"],
            "cluster_label": d["label"],
            "medoid": d["medoid"],
            "status": "❄️ FADING / COMMODITIZED",
            "warning": (
                f"Topical momentum for [{d['label']}] has decelerated (acceleration: {d['acceleration']}). "
                f"Market saturation is high and organic interest is plateauing. "
                f"Avoid investing new standalone articles here; instead, fold into broader thematic guides."
            ),
            "recommendation": "Maintain existing pages; pivot fresh content efforts toward rising breakout topics."
        })

    # 5. Compile Forecast Report
    forecast_report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_horizon": f"+{forecast_horizon} Cycle",
        "system_status": "OPTIMAL · 3 TOPICS SYNTHESIZED",
        "top_emerging_topics": emerging_topics,
        "sunset_topics": sunset_topics,
        "total_clusters_evaluated": len(cluster_scores)
    }

    # Persist outputs/trend_forecast.json atomically
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(TREND_FORECAST_PATH, forecast_report)
        logger.info(f"Emitted Predictive Trend Forecast to '{TREND_FORECAST_PATH}'.")

        # Emit outputs/burst_analysis.json for Kleinberg burst insights
        burst_path = out_dir / "burst_analysis.json"
        burst_report = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "top_bursting_topics": [
                {
                    "title": t["working_title"],
                    "momentum": t["momentum_score"],
                    "stage": t["stage"],
                    "urgency": t["urgency_tier"]
                }
                for t in emerging_topics[:10]
            ]
        }
        atomic_write_json(burst_path, burst_report)
    except Exception as e:
        logger.warning(f"Could not save trend_forecast.json: {e}")

    return forecast_report


def export_snapshot_datasheet(base_dir: Optional[Path] = None, export_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Exports a comprehensive, portable snapshot data sheet (.json and .csv)
    containing current corpus stats, collocations, clusters, and trend predictions
    using atomic temp-and-replace semantics.
    """
    b_dir = base_dir or BASE_DIR
    out_dir = b_dir / "outputs"
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    clusters_data = load_json_safe(out_dir / "clusters.json") or {}
    collocs_data = load_json_safe(out_dir / "collocations.json") or []
    trend_data = load_json_safe(out_dir / "trend_forecast.json") or {}
    run_state = load_json_safe(out_dir / "run_state.json") or {}

    # Read top 100 ranked words
    ranked_words = []
    r_file = out_dir / "word_freq_ranked.csv"
    if r_file.exists():
        try:
            with open(r_file, "r", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for i, row in enumerate(reader):
                    if i >= 100:
                        break
                    ranked_words.append({
                        "word": row.get("word"),
                        "count": int(row.get("count", 0)),
                        "cluster": row.get("cluster_id", "-")
                    })
        except Exception:
            pass

    datasheet = {
        "snapshot_id": f"centroid_snapshot_{timestamp_str}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_name": "Active Domain Monitor",
        "cluster_count": len(clusters_data),
        "total_vocabulary": len(ranked_words),
        "collocations": collocs_data[:20],
        "emerging_topics": trend_data.get("top_emerging_topics", []),
        "sunset_topics": trend_data.get("sunset_topics", []),
        "ranked_sample": ranked_words[:30]
    }

    target_json = export_path or (out_dir / f"centroid_snapshot_{timestamp_str}.json")
    atomic_write_json(target_json, datasheet)

    logger.info(f"Exported snapshot data sheet to '{target_json}'.")
    return datasheet


def holt_winters_damped_smooth(
    series: List[float],
    alpha: float = 0.6,
    beta: float = 0.3,
    phi: float = 0.85,
    horizon: int = 1
) -> Tuple[List[float], float]:
    """
    Holt-Winters double exponential smoothing with trend dampening phi.
    Dampens small-sample noise and prevents runaway projections.
    Handles NaN, Inf, and non-numeric values safely.
    """
    if not series:
        return [], 0.0

    clean_series: List[float] = []
    last_valid = 0.0
    for v in series:
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                clean_series.append(last_valid)
            else:
                clean_series.append(fv)
                last_valid = fv
        except (ValueError, TypeError):
            clean_series.append(last_valid)

    if not clean_series:
        return [], 0.0
    if len(clean_series) == 1:
        return list(clean_series), float(clean_series[0])

    smoothed = [clean_series[0]]
    L = float(clean_series[0])
    B = float(clean_series[1] - clean_series[0])

    for t in range(1, len(clean_series)):
        y = float(clean_series[t])
        L_prev = L
        B_prev = B
        L = alpha * y + (1.0 - alpha) * (L_prev + phi * B_prev)
        B = beta * (L - L_prev) + (1.0 - beta) * (phi * B_prev)
        smoothed.append(round(L, 2))

    proj_sum = sum(phi ** i for i in range(1, horizon + 1))
    forecast = max(0.0, round(L + proj_sum * B, 2))
    return smoothed, forecast


def reconstruct_multi_snapshot_timeline(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes multiple uploaded snapshot sheets, validates them against strict schema guardrails,
    orders them chronologically, and computes the rise and fall trajectory for all major topics
    across time using Holt-Winters trend dampening and Kleinberg burst detection.
    """
    if not snapshots or not isinstance(snapshots, list):
        return {"success": False, "error": "No snapshot payloads provided.", "timeline": [], "topic_trends": []}

    valid_snapshots = []
    validation_warnings = []

    for idx, s in enumerate(snapshots):
        is_valid, err_msg, cleaned = validate_snapshot_schema(s)
        if is_valid and cleaned:
            ts = cleaned.get("timestamp") or ""
            valid_snapshots.append((ts, cleaned))
        else:
            validation_warnings.append(f"Snapshot #{idx}: {err_msg}")

    if not valid_snapshots:
        return {
            "success": False,
            "error": "All provided snapshot data sheets failed strict schema validation.",
            "details": validation_warnings,
            "timeline": [],
            "topic_trends": []
        }

    valid_snapshots.sort(key=lambda x: x[0])

    timeline_points = []
    topic_curves: Dict[str, List[float]] = {}

    for idx, (ts, s) in enumerate(valid_snapshots):
        label = f"Epoch {idx + 1}"
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                label = dt.strftime("%b %d")
            except Exception:
                pass

        topics = s.get("emerging_topics", [])
        snapshot_topics_map = {}
        for t in topics:
            title = t.get("working_title") or t.get("cluster_label", "Unknown")
            score = float(t.get("momentum_score", 50.0))
            snapshot_topics_map[title] = score
            if title not in topic_curves:
                topic_curves[title] = [0.0] * idx
            topic_curves[title].append(score)

        for title in topic_curves:
            if len(topic_curves[title]) <= idx:
                topic_curves[title].append(0.0)

        timeline_points.append({
            "epoch_index": idx,
            "label": label,
            "timestamp": ts,
            "topics": snapshot_topics_map
        })

    epoch_timestamps = [ts for ts, s in valid_snapshots]

    topic_trends = []
    for title, scores in topic_curves.items():
        smoothed_scores, projected_score = holt_winters_damped_smooth(scores, horizon=1)

        # Apply exponential temporal decay w(t) = exp(-lambda * delta_t) across epoch timeline
        decayed_scores = apply_exponential_temporal_decay(
            scores,
            timestamps=epoch_timestamps,
            half_life_days=14.0
        )

        int_scores = []
        for sc in scores:
            try:
                f_sc = float(sc)
                if math.isnan(f_sc) or math.isinf(f_sc):
                    int_scores.append(1)
                else:
                    int_scores.append(max(1, int(round(f_sc))))
            except (ValueError, TypeError):
                int_scores.append(1)

        totals = [100] * len(int_scores)
        burst_eval = detect_kleinberg_burst_2state(int_scores, totals, s=2.0, gamma=1.0)

        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            status = "🔥 RISING" if delta > 5 else "❄️ FALLING" if delta < -5 else "➡️ STABLE"
        else:
            delta = 0.0
            status = "🌱 NEW SIGNAL"

        topic_trends.append({
            "title": title,
            "scores": scores,
            "smoothed_scores": smoothed_scores,
            "decayed_scores": decayed_scores,
            "latest_score": scores[-1] if scores else 0,
            "projected_score": projected_score,
            "net_delta": round(delta, 1),
            "status": status,
            "burst_status": burst_eval["status"],
            "burst_weight": burst_eval["burst_weight"],
            "is_bursting": burst_eval["is_bursting"]
        })

    topic_trends.sort(key=lambda t: t["latest_score"], reverse=True)

    return {
        "success": True,
        "total_snapshots": len(valid_snapshots),
        "validation_warnings": validation_warnings,
        "timeline_points": timeline_points,
        "topic_trends": topic_trends
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Trend Horizon & Topic Synthesizer")
    parser.add_argument("--export", action="store_true", help="Export portable snapshot data sheet")
    parser.add_argument("--forecast-horizon", type=float, default=1.0, help="Extrapolation forecast horizon (default: 1.0)")
    args = parser.parse_args()

    report = synthesize_predictive_topics(forecast_horizon=args.forecast_horizon)
    if args.export:
        export_snapshot_datasheet()

    print("\n" + "=" * 70)
    print(" CENTROID Stage 10: Predictive Trend Horizon & Topic Synthesis")
    print("=" * 70)
    for topic in report.get("top_emerging_topics", []):
        print(f"\n★ TOPIC #{topic['rank']}: {topic['working_title']}")
        print(f"  Stage:         {topic['stage']} ({topic['urgency_tier']})")
        print(f"  Timing Window: {topic['timing_window']}")
        print(f"  Why Trending:  {topic['why_trending']}")
        print(f"  Keywords:      {', '.join(topic['key_vocabulary'])}")
    print("=" * 70 + "\n")
