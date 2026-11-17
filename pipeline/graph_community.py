#!/usr/bin/env python3
"""
CENTROID Pipeline — Co-Occurrence Graph Community Detection & Bridge Centrality
Location: pipeline/graph_community.py

Description:
  Constructs an undirected co-occurrence graph between mined terms and collocations
  and executes pure graph algorithms without AI APIs:
    1. Louvain Modularity Community Partitioning to isolate organic topic clusters.
    2. PageRank Centrality & Degree Centrality via power iteration.
    3. Bridge Concept Detection via Participation Coefficient to identify inter-domain bridges.
"""

import sys
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import atomic_write_json
except ImportError:
    from common import atomic_write_json


def build_cooccurrence_graph(
    sentences: List[List[str]],
    top_words: List[str],
    window_size: int = 5,
    min_edge_weight: int = 1
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Builds an undirected, weighted co-occurrence graph between top vocabulary words
    over a sliding window.
    """
    word_set = set(top_words)
    adj: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    half_window = max(1, window_size // 2)

    for sentence in sentences:
        s_len = len(sentence)
        for i, word in enumerate(sentence):
            if word in word_set:
                end = min(s_len, i + half_window + 1)
                for j in range(i + 1, end):
                    context_word = sentence[j]
                    if context_word in word_set and context_word != word:
                        # Undirected edge: increment both directions exactly once
                        adj[word][context_word] += 1
                        adj[context_word][word] += 1

    # Filter by minimum edge weight and prune isolated nodes if requested
    filtered_adj: Dict[str, Dict[str, int]] = defaultdict(dict)
    active_nodes_set: Set[str] = set()

    for u in top_words:
        if u in adj:
            for v, weight in adj[u].items():
                if weight >= min_edge_weight:
                    filtered_adj[u][v] = weight
                    active_nodes_set.add(u)
                    active_nodes_set.add(v)

    # Return active nodes preserving top_words ordering, plus any connected nodes
    node_list = [w for w in top_words if w in active_nodes_set]
    if not node_list and top_words:
        # Fallback if no co-occurrences meet threshold
        node_list = list(top_words)

    return node_list, filtered_adj


def compute_pagerank(
    nodes: List[str],
    adj: Dict[str, Dict[str, int]],
    damping: float = 0.85,
    max_iter: int = 30,
    tol: float = 1e-6
) -> Dict[str, float]:
    """
    Computes PageRank centrality on the undirected weighted graph using power iteration.
    """
    N = len(nodes)
    if N == 0:
        return {}
    if N == 1:
        return {nodes[0]: 1.0}

    # Node weighted degree
    degree: Dict[str, float] = {}
    for u in nodes:
        degree[u] = float(sum(adj.get(u, {}).values()))

    # Initialize uniform distribution
    pr = {u: 1.0 / N for u in nodes}
    teleport = (1.0 - damping) / N

    for _ in range(max_iter):
        next_pr = {u: teleport for u in nodes}
        dangling_sum = sum(pr[u] for u in nodes if degree[u] <= 0)

        for u in nodes:
            deg_u = degree[u]
            if deg_u > 0:
                p_u = pr[u]
                for v, w in adj.get(u, {}).items():
                    if v in next_pr:
                        next_pr[v] += damping * (p_u * (w / deg_u))

        if dangling_sum > 0:
            dangling_contrib = damping * (dangling_sum / N)
            for u in nodes:
                next_pr[u] += dangling_contrib

        # Check convergence
        diff = sum(abs(next_pr[u] - pr[u]) for u in nodes)
        pr = next_pr
        if diff < tol:
            break

    # Normalize sum to 1.0
    total = sum(pr.values()) or 1.0
    return {u: round(pr[u] / total, 6) for u in nodes}


def detect_louvain_communities(
    nodes: List[str],
    adj: Dict[str, Dict[str, int]],
    max_iter: int = 20
) -> Tuple[Dict[str, int], float]:
    """
    Deterministic Louvain Modularity Partitioning for community detection.
    Optimizes Newman-Girvan modularity Q.
    Returns: (node_to_community_map, modularity_Q)
    """
    N = len(nodes)
    if N == 0:
        return {}, 0.0
    if N == 1:
        return {nodes[0]: 0}, 1.0

    # Calculate total weight m and degrees
    deg: Dict[str, float] = {u: float(sum(adj.get(u, {}).values())) for u in nodes}
    total_m2 = sum(deg.values())  # 2 * m
    if total_m2 <= 0:
        # No edges; each node in isolated community
        return {u: i for i, u in enumerate(nodes)}, 0.0

    # Each node initially in its own community
    community = {u: i for i, u in enumerate(nodes)}
    comm_total_deg: Dict[int, float] = {i: deg[u] for i, u in enumerate(nodes)}

    # Local modularity optimization pass
    for _ in range(max_iter):
        moves = 0
        for u in nodes:
            k_u = deg[u]
            if k_u <= 0:
                continue

            c_old = community[u]

            # Weights from u to neighboring communities
            comm_weights: Dict[int, float] = defaultdict(float)
            for v, w in adj.get(u, {}).items():
                if v in community:
                    comm_weights[community[v]] += float(w)

            # Modularity gain of removing u from c_old
            comm_total_deg[c_old] -= k_u
            best_c = c_old
            best_gain = 0.0

            for c, k_u_in in comm_weights.items():
                tot_c = comm_total_deg[c]
                # Modularity delta
                gain = (k_u_in / total_m2) - (k_u * tot_c / (total_m2 ** 2))
                if gain > best_gain:
                    best_gain = gain
                    best_c = c

            community[u] = best_c
            comm_total_deg[best_c] += k_u
            if best_c != c_old:
                moves += 1

        if moves == 0:
            break

    # Compact community IDs to 0 .. K-1 sorted by community size
    comm_members: Dict[int, List[str]] = defaultdict(list)
    for u, c in community.items():
        comm_members[c].append(u)

    sorted_comms = sorted(comm_members.values(), key=lambda mems: len(mems), reverse=True)
    compact_comm: Dict[str, int] = {}
    for new_id, mems in enumerate(sorted_comms):
        for u in mems:
            compact_comm[u] = new_id

    # Compute final modularity Q
    q = 0.0
    for u in nodes:
        c_u = compact_comm[u]
        k_u = deg[u]
        for v, w in adj.get(u, {}).items():
            if v in compact_comm and compact_comm[v] == c_u:
                q += w - (k_u * deg[v] / total_m2)
    final_q = round(q / total_m2, 4) if total_m2 > 0 else 0.0

    return compact_comm, final_q


def detect_bridge_concepts(
    nodes: List[str],
    adj: Dict[str, Dict[str, int]],
    communities: Dict[str, int],
    pagerank: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Identifies bridge concepts (terms that span multiple topic clusters)
    via Participation Coefficient:
      P_v = 1 - sum_{C} (k_{v, C} / k_v)^2
    """
    bridge_records = []

    for u in nodes:
        neighbors = adj.get(u, {})
        k_u = sum(neighbors.values())
        if k_u <= 0:
            continue

        c_u = communities.get(u, -1)

        # Distribute degree across communities
        comm_deg: Dict[int, int] = defaultdict(int)
        for v, w in neighbors.items():
            c_v = communities.get(v, -1)
            comm_deg[c_v] += w

        # Participation coefficient
        sum_sq = sum((w_c / k_u) ** 2 for w_c in comm_deg.values())
        participation_coeff = max(0.0, 1.0 - sum_sq)

        pr = pagerank.get(u, 0.0)
        # Bridge score combines participation coefficient with PageRank authority
        bridge_score = participation_coeff * (pr * 1000.0)

        # Count external connected communities
        external_comms = [c for c in comm_deg if c != c_u]

        if participation_coeff > 0.1 and len(comm_deg) > 1:
            bridge_records.append({
                "word": u,
                "home_community_id": c_u,
                "bridge_score": round(bridge_score, 3),
                "participation_coefficient": round(participation_coeff, 3),
                "pagerank": round(pr, 6),
                "total_degree": k_u,
                "spanning_communities_count": len(comm_deg),
                "connected_communities": [int(c) for c in sorted(comm_deg.keys())]
            })

    bridge_records.sort(key=lambda x: (-x["bridge_score"], -x["pagerank"]))
    return bridge_records


def merge_collocations_in_sentences(
    sentences: List[List[str]],
    phrases: List[str]
) -> List[List[str]]:
    """
    Merges sequences of tokens that form known multi-word collocations
    into single combined tokens so they participate in the co-occurrence graph.
    """
    if not phrases or not sentences:
        return sentences

    phrase_tokens = [p.split() for p in sorted(phrases, key=lambda x: len(x.split()), reverse=True) if len(p.split()) > 1]
    if not phrase_tokens:
        return sentences

    merged_sentences: List[List[str]] = []
    for sentence in sentences:
        s_len = len(sentence)
        new_sent: List[str] = []
        i = 0
        while i < s_len:
            matched = False
            for pt in phrase_tokens:
                k = len(pt)
                if i + k <= s_len and sentence[i:i + k] == pt:
                    new_sent.append(" ".join(pt))
                    i += k
                    matched = True
                    break
            if not matched:
                new_sent.append(sentence[i])
                i += 1
        merged_sentences.append(new_sent)
    return merged_sentences


def run_cooccurrence_graph_pipeline(
    sentences: List[List[str]],
    top_words: List[str],
    collocations: Optional[List[Dict[str, Any]]] = None,
    base_dir: Optional[Path] = None,
    window_size: int = 5
) -> Dict[str, Any]:
    """
    Executes full co-occurrence graph analysis and saves outputs/cooccurrence_graph.json.
    """
    b_dir = base_dir or Path.cwd()
    out_dir = b_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally integrate multi-word collocations into graph terms
    combined_terms = list(top_words)
    colloc_phrases: List[str] = []
    if collocations:
        for c in collocations[:30]:
            p = c.get("phrase")
            if p and p not in combined_terms:
                combined_terms.append(p)
                colloc_phrases.append(p)

    graph_sentences = merge_collocations_in_sentences(sentences, colloc_phrases) if colloc_phrases else sentences
    nodes, adj = build_cooccurrence_graph(graph_sentences, combined_terms, window_size=window_size)
    communities, modularity_q = detect_louvain_communities(nodes, adj)
    pagerank = compute_pagerank(nodes, adj)
    bridge_concepts = detect_bridge_concepts(nodes, adj, communities, pagerank)

    # Group community summaries
    comm_groups: Dict[int, List[str]] = defaultdict(list)
    for u, c_id in communities.items():
        comm_groups[c_id].append(u)

    communities_summary = []
    for c_id, members in sorted(comm_groups.items(), key=lambda item: len(item[1]), reverse=True):
        # Order members by PageRank
        sorted_members = sorted(members, key=lambda m: pagerank.get(m, 0.0), reverse=True)
        top_anchor = sorted_members[0] if sorted_members else f"Topic {c_id}"
        communities_summary.append({
            "community_id": c_id,
            "label": f"Topic: {top_anchor}",
            "core_medoid": top_anchor,
            "size": len(members),
            "top_words": sorted_members[:15],
            "highest_pagerank": pagerank.get(top_anchor, 0.0)
        })

    graph_payload = {
        "graph_stats": {
            "total_nodes": len(nodes),
            "total_edges": sum(len(neighbors) for neighbors in adj.values()) // 2,
            "modularity_q": modularity_q,
            "communities_count": len(communities_summary),
            "bridge_concepts_count": len(bridge_concepts)
        },
        "communities": communities_summary,
        "bridge_concepts": bridge_concepts[:25],
        "pagerank_top_20": sorted([{"word": w, "pagerank": pr} for w, pr in pagerank.items()], key=lambda x: -x["pagerank"])[:20]
    }

    target_path = out_dir / "cooccurrence_graph.json"
    atomic_write_json(target_path, graph_payload)
    return graph_payload
