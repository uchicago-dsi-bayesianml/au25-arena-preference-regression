import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

def analyze_inertia(
    emb_array: np.ndarray,
    k_min: int = 5,
    k_max: int = 50,
    step: int = 5,
    random_state: int = 42
):
    inertias = []
    k_values = range(k_min, k_max + 1, step)
    
    print(f"Analyzing inertia for K in {list(k_values)}...")

    for k in tqdm(k_values, desc="Inertia Sweep"):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
            max_iter=2000,
        )
        kmeans.fit(emb_array)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='--')
    plt.title('Elbow Method: Inertia vs Number of Clusters')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.grid(True)
    plt.show()
    return k_values, inertias

def get_cluster_archetypes(df, emb_array, kmeans, column, n_samples=5):
    distances = kmeans.transform(emb_array)
    cluster_archetypes = {}
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue

        cluster_distances = distances[cluster_indices, cluster_id]
        sorted_subset_indices = np.argsort(cluster_distances)
        sorted_original_indices = cluster_indices[sorted_subset_indices]
        
        buffer_size = min(len(sorted_original_indices), n_samples * 10)
        candidate_indices = sorted_original_indices[:buffer_size]
        candidate_texts = df.iloc[candidate_indices][column].tolist()
        
        unique_texts = []
        seen = set()
        
        for text in candidate_texts:
            clean_text = str(text).strip()
            if clean_text not in seen and clean_text != "":
                unique_texts.append(clean_text)
                seen.add(clean_text)
            
            if len(unique_texts) >= n_samples:
                break
        
        cluster_archetypes[cluster_id] = unique_texts
        
    return cluster_archetypes

def auto_label_clusters(archetypes_dict, client, model="gpt-5-mini"):
    PROMPT_TEMPLATE = """
    You are a topic labeling assistant. You will be given 5 representative snippets from a conversation cluster.
    Your goal is to provide a single, short, descriptive label (max 4 words) that captures the common theme.
    Output ONLY the label. Do not use quotes or introductory text.

    Examples:
    - Python Debugging
    - Politics & Election
    - Creative Writing

    Snippets:
    {snippets}

    Label:
    """
    cluster_names = {}
    
    print("Generating cluster labels...")
    for cid, texts in tqdm(archetypes_dict.items()):
        examples_str = "\n---\n".join(texts)
        prompt = PROMPT_TEMPLATE.format(snippets=examples_str)
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "minimal"},
            )
            if hasattr(response, 'output_text'):
                label = response.output_text
            else:
                label = str(response)

            cluster_names[cid] = label.strip().replace('"', '')
        except Exception as e:
            print(f"Error labeling cluster {cid}: {e}")
            cluster_names[cid] = f"Cluster {cid}"
            
    return cluster_names