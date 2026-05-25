import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.decomposition import PCA


def greedy_clustering_adaptive(features, threshold, metric='cosine', use_pca=False, pca_dim=64):
    """Adaptive greedy clustering with medoid refinement.

    Assigns each sample to the nearest cluster center if within threshold,
    otherwise creates a new cluster. After clustering, maps virtual centers
    to real samples (medoids).

    Args:
        features: (N, D) feature matrix
        threshold:
            - metric='euclidean': distance threshold (smaller = stricter)
            - metric='cosine': similarity threshold (larger = stricter, e.g. 0.95~0.99)
        metric: 'euclidean' or 'cosine'
        use_pca: whether to apply PCA dimensionality reduction
        pca_dim: PCA target dimension

    Returns:
        final_labels: (N,) medoid ID for each sample
        medoid_indices: (K,) list of all medoid indices
    """
    if features.dtype != np.float32:
        features = features.astype(np.float32)

    original_dim = features.shape[1]
    process_features = features

    # PCA preprocessing
    if use_pca and original_dim > pca_dim:
        print(f"=> [Preprocess] PCA Reducing dimension: {original_dim} -> {pca_dim}")
        pca = PCA(n_components=pca_dim)
        process_features = pca.fit_transform(features)

        if metric == 'cosine':
            print("=> [Preprocess] Re-normalizing after PCA...")
            norm = np.linalg.norm(process_features, axis=1, keepdims=True)
            process_features = process_features / (norm + 1e-10)

    num_samples = process_features.shape[0]

    # Initialize storage
    centers_sum = []
    centers_count = []
    centers_curr = []

    temp_labels = np.full(num_samples, -1, dtype=int)

    print(f"=> [Adaptive Clustering] Start... (N={num_samples}, Metric={metric}, Threshold={threshold})")

    # Main clustering loop
    for i in tqdm(range(num_samples)):
        current_vec = process_features[i]

        if len(centers_curr) == 0:
            centers_sum.append(current_vec)
            centers_count.append(1)
            centers_curr.append(current_vec)
            temp_labels[i] = 0
            continue

        C_mat = np.array(centers_curr)

        if metric == 'cosine':
            sims = np.dot(C_mat, current_vec)
            best_idx = np.argmax(sims)
            is_match = (sims[best_idx] >= threshold)
        else:
            dists_sq = np.sum((C_mat - current_vec)**2, axis=1)
            best_idx = np.argmin(dists_sq)
            is_match = (dists_sq[best_idx] <= threshold**2)

        if is_match:
            temp_labels[i] = best_idx
            centers_sum[best_idx] = centers_sum[best_idx] + current_vec
            centers_count[best_idx] += 1

            new_center = centers_sum[best_idx] / centers_count[best_idx]

            if metric == 'cosine':
                new_center = new_center / (np.linalg.norm(new_center) + 1e-10)

            centers_curr[best_idx] = new_center
        else:
            new_id = len(centers_curr)
            temp_labels[i] = new_id
            centers_sum.append(current_vec)
            centers_count.append(1)
            centers_curr.append(current_vec)

    num_clusters = len(centers_curr)
    print(f"=> Clustering phase done. Generated {num_clusters} temporary clusters.")

    # Post-processing: find medoids (real representative samples)
    print("=> [Post-processing] Mapping virtual centers to real samples (Medoids)...")

    final_labels = np.zeros_like(temp_labels)
    medoid_indices = []

    for cluster_id in tqdm(range(num_clusters)):
        member_indices = np.where(temp_labels == cluster_id)[0]

        if len(member_indices) == 1:
            medoid_id = member_indices[0]
        else:
            member_vecs = process_features[member_indices]
            cluster_mean = np.mean(member_vecs, axis=0)

            if metric == 'cosine':
                cluster_mean = cluster_mean / (np.linalg.norm(cluster_mean) + 1e-10)
                scores = np.dot(member_vecs, cluster_mean)
                best_local_idx = np.argmax(scores)
            else:
                dists = np.sum((member_vecs - cluster_mean)**2, axis=1)
                best_local_idx = np.argmin(dists)

            medoid_id = member_indices[best_local_idx]

        final_labels[member_indices] = medoid_id
        medoid_indices.append(medoid_id)

    print(f"=> Medoid mapping complete. Total unique representatives: {len(medoid_indices)}")

    return final_labels, medoid_indices


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.cluster <results_dir>")
        print("  Expects features.npy and filenames.txt in <results_dir>")
        sys.exit(1)

    task_root = sys.argv[1]
    feature_path = os.path.join(task_root, "features.npy")

    print(f"=> Loading features from {feature_path}...")
    features = np.load(feature_path)

    METRIC = 'cosine'
    THRESHOLD = 0.9
    USE_PCA = False
    PCA_DIM = 64

    t0 = time.time()
    labels, representatives = greedy_clustering_adaptive(
        features,
        threshold=THRESHOLD,
        metric=METRIC,
        use_pca=USE_PCA,
        pca_dim=PCA_DIM
    )
    print(f"Total Process Time: {time.time()-t0:.2f}s")

    np.savetxt(os.path.join(task_root, "clusters.txt"), labels, fmt='%d')
