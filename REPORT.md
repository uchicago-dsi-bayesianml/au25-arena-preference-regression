# Bayes Final Project – Conversation Analysis

This repo contains a small toolkit for analyzing LLM chat logs using OpenAI models, embeddings, and clustering. It is organized around a few worker utilities that can be called from notebooks or scripts.

## Info
- Author: Yusen Wu, Yifei Zhou
- Final Project for ADSP 32014 Bayesian Machine Learning with Generative AI Applications

## Project layout
- `main.ipynb` – example / scratch notebook to drive experiments.
- `utils.py` – helpers for turning raw conversations into message lists and computing basic conversation-level features (e.g., number of turns, token lengths).
- `format_workers.py` – classifies assistant formatting (code blocks, bullets, headings) using an OpenAI model and runs the classification in parallel over a DataFrame.
- `embedding_workers.py` – computes and caches text embeddings for conversations, then runs KMeans topic clustering over the embeddings.
- `cluster_workers.py` – downstream cluster utilities: inertia sweep (elbow plot), selecting cluster archetype examples, and auto-labeling clusters via an OpenAI model.
- `out/`, `results/` – output folders for intermediate artifacts (embeddings, clustered DataFrames, plots, etc.).

## Requirements
- Python 3.10+ (recommended).
- Libraries: `openai`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`, `nltk`.
- An OpenAI API key available as `OPENAI_API_KEY` in your environment.

Install dependencies (example):

```bash
pip install openai numpy pandas scikit-learn tqdm matplotlib nltk
```

## Typical workflow
1. Load your conversation dataset into a `pandas.DataFrame`.
2. Use `format_workers.run_parallel_format(...)` to compute formatting flags for each conversation side.
3. Use `embedding_workers.compute_or_load_embeddings(...)` and `embedding_workers.run_topic_clustering(...)` to get topic clusters and embeddings.
4. Use `cluster_workers.analyze_inertia(...)`, `get_cluster_archetypes(...)`, and `auto_label_clusters(...)` to inspect and label clusters.
5. Drive end-to-end experiments from `main.ipynb` or your own script.

