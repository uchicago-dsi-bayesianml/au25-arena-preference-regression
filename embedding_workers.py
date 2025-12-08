import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from openai import OpenAI

client = OpenAI()
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 2048


def compute_or_load_embeddings(
    df: pd.DataFrame,
    text_column: str,
    embed_cache_path: str = "conv_embeddings.npy",
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    n_rows = len(df)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    if os.path.exists(embed_cache_path):
        print(f"[analyze_workers] Loading cached embeddings from {embed_cache_path}...")
        emb_array = np.load(embed_cache_path)
        
        if emb_array.shape[0] == n_rows:
            return emb_array
        else:
            print(
                f"[analyze_workers] Cache mismatch: Cache has {emb_array.shape[0]} rows, "
                f"DF has {n_rows}. Recomputing..."
            )

    print(f"[analyze_workers] Computing embeddings on column '{text_column}'...")
    
    texts = df[text_column].fillna("").astype(str).tolist()

    nonempty_indices = [i for i, t in enumerate(texts) if t.strip()]
    nonempty_texts = [texts[i] for i in nonempty_indices]

    emb_list = [None] * n_rows

    if not nonempty_texts:
        print("[analyze_workers] Warning: All text entries are empty. Returning zero matrix.")
        return np.zeros((n_rows, 1), dtype=float) 

    for start in tqdm(range(0, len(nonempty_texts), batch_size), desc="Embedding"):
        end = start + batch_size
        batch_texts = nonempty_texts[start:end]

        try:
            resp = client.embeddings.create(
                model=embed_model,
                input=batch_texts,
            )
            batch_embs = [d.embedding for d in resp.data]

            for j, emb in enumerate(batch_embs):
                orig_idx = nonempty_indices[start + j]
                emb_list[orig_idx] = emb
                
        except Exception as e:
            print(f"Error generating embeddings for batch {start}-{end}: {e}")
            raise e

    valid_embs = [e for e in emb_list if e is not None]
    if valid_embs:
        dim = len(valid_embs[0])
    else:
        dim = 1536

    emb_array = np.array([
        e if e is not None else np.zeros(dim, dtype=float)
        for e in emb_list
    ])

    np.save(embed_cache_path, emb_array)
    print(f"[analyze_workers] Saved embeddings to {embed_cache_path}. Shape: {emb_array.shape}")
    
    return emb_array


def run_kmeans(
    emb_array: np.ndarray,
    n_clusters: int = 30,
    random_state: int = 42,
):
    print(f"[analyze_workers] Running KMeans with K={n_clusters}...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        max_iter=2000,
    )
    cluster_ids = kmeans.fit_predict(emb_array)
    return cluster_ids, kmeans


def run_topic_clustering(
    df: pd.DataFrame,
    text_column: str,
    embed_cache_path: str = "conv_embeddings.npy",
    embed_model: str = DEFAULT_EMBED_MODEL,
    n_clusters: int = 30,
    save_df_path: str | None = None,
    save_embeddings_in_df: bool = True,
):
    emb_array = compute_or_load_embeddings(
        df=df,
        text_column=text_column,
        embed_cache_path=embed_cache_path,
        embed_model=embed_model
    )

    cluster_ids, kmeans = run_kmeans(
        emb_array=emb_array,
        n_clusters=n_clusters
    )

    df["topic_cluster"] = cluster_ids

    if save_embeddings_in_df:
        df["topic_embedding"] = [emb_array[i].tolist() for i in range(emb_array.shape[0])]

    print("[analyze_workers] Cluster counts:")
    print(df["topic_cluster"].value_counts().sort_index())

    if save_df_path:
        df.to_pickle(save_df_path)
        print(f"[analyze_workers] Saved results to {save_df_path}")

    return df, emb_array, kmeans