# CENTROID Example Workflow

This directory provides a minimal, self-contained, reproducible example demonstrating CENTROID's end-to-end analytical workflow across distinct subject domains (quantum computing, marine biology, culinary gastronomy).

---

## Directory Contents

- `sample_input/`:
  - `article_quantum_computing.txt`: Synthetic technical overview of qubits, superposition, and error correction.
  - `article_marine_ecosystems.txt`: Overview of coral reefs, trophic webs, and ocean conservation.
  - `article_culinary_gastronomy.txt`: Overview of fermentation, Maillard browning, and molecular emulsions.

---

## Running the Example

To execute CENTROID against these sample inputs:

```bash
# 1. Copy sample inputs into the active inputs/docs directory
# Windows (PowerShell):
Copy-Item examples\sample_input\*.txt inputs\docs\

# macOS / Linux:
cp examples/sample_input/*.txt inputs/docs/

# 2. Run the pipeline with k=3 clusters and dashboard generation
python run.py --top 500 --clusters 3 --dashboard --force
```

---

## Expected Results

When configured with $k=3$ clusters, CENTROID's Word2Vec and KMeans engine naturally segments the three domains into distinct topological semantic clusters:
1. **Cluster 1 (Quantum Domain)**: Anchored by terms such as `quantum`, `qubits`, `superposition`, `hardware`.
2. **Cluster 2 (Marine Domain)**: Anchored by terms such as `marine`, `coral`, `ocean`, `ecosystems`.
3. **Cluster 3 (Culinary Domain)**: Anchored by terms such as `culinary`, `flavor`, `gastronomy`, `fermentation`.

Analytical outputs are generated in `outputs/`:
- `outputs/REPORT.md`: Comprehensive Markdown report.
- `outputs/dashboard.html`: Interactive single-file visualization dashboard.
- `outputs/clusters.json`: Structured vector clustering definitions.
- `outputs/word_freq_ranked.csv`: Ranked word scores and cluster assignments.
