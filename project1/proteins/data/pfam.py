"""SwissProt-Pfam -> ESM-2 latent tensors with cluster-based splitting.

Representation:
    z      [B, L, 320]  ESM-2 per-residue hidden states, standardized
    label  [B]          Pfam family index (0, 1, 2 for top 3 families)
    seq    [B]          original sequence strings (for novelty checking)

Pipeline:
    1. Load DanielHesslow/SwissProt-Pfam from HuggingFace
    2. Pick top N_FAMILIES most populated Pfam families
    3. Filter to sequences within length range, crop/pad to SEQ_LEN
    4. Deduplicate exact sequences
    5. Cluster by k-mer fingerprint similarity to avoid homolog leakage
    6. Split clusters into train/val/test
    7. Encode with frozen ESM-2 (facebook/esm2_t6_8M_UR50D)
    8. Standardize latents (subtract mean, divide by std)
    9. Cache to .pt
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
import numpy as np

SEQ_LEN = 64            # fixed sequence length
HIDDEN_DIM = 320        # ESM-2 8M hidden dimension
N_FAMILIES = 3          # top N Pfam families to use
MIN_LEN = 32            # minimum sequence length before padding
MAX_LEN = 150           # maximum sequence length before cropping
KMER_K = 3              # k-mer size for clustering
CLUSTER_THRESH = 0.5    # k-mer Jaccard threshold for same cluster
SPLIT = (0.8, 0.1)      # train, val fractions; remainder is test


def get_kmers(seq, k=KMER_K):
    """Get set of k-mers from sequence."""
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))


def kmer_jaccard(kmers1, kmers2):
    """Jaccard similarity between two k-mer sets."""
    if not kmers1 or not kmers2:
        return 0.0
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union if union > 0 else 0.0


def greedy_cluster(sequences, threshold=CLUSTER_THRESH):
    """Greedy clustering by k-mer similarity.

    Returns list of cluster IDs, one per sequence.
    """
    kmers_list = [get_kmers(s) for s in sequences]
    n = len(sequences)
    cluster_ids = [-1] * n
    cluster_reps = []  # (cluster_id, representative_kmers)

    for i in range(n):
        best_cluster = -1
        best_sim = 0.0

        for cid, rep_kmers in cluster_reps:
            sim = kmer_jaccard(kmers_list[i], rep_kmers)
            if sim > best_sim and sim >= threshold:
                best_sim = sim
                best_cluster = cid

        if best_cluster == -1:
            # Start new cluster
            new_cid = len(cluster_reps)
            cluster_ids[i] = new_cid
            cluster_reps.append((new_cid, kmers_list[i]))
        else:
            cluster_ids[i] = best_cluster

    return cluster_ids


def split_by_clusters(cluster_ids, seed=0):
    """Split data indices by clusters to avoid homolog leakage."""
    rng = np.random.RandomState(seed)
    unique_clusters = list(set(cluster_ids))
    rng.shuffle(unique_clusters)

    n_clusters = len(unique_clusters)
    n_train = int(SPLIT[0] * n_clusters)
    n_val = int(SPLIT[1] * n_clusters)

    train_clusters = set(unique_clusters[:n_train])
    val_clusters = set(unique_clusters[n_train:n_train + n_val])
    test_clusters = set(unique_clusters[n_train + n_val:])

    train_idx = [i for i, c in enumerate(cluster_ids) if c in train_clusters]
    val_idx = [i for i, c in enumerate(cluster_ids) if c in val_clusters]
    test_idx = [i for i, c in enumerate(cluster_ids) if c in test_clusters]

    return train_idx, val_idx, test_idx


def pad_or_crop(seq, target_len=SEQ_LEN):
    """Pad with X or crop sequence to target length."""
    if len(seq) >= target_len:
        # Center crop
        start = (len(seq) - target_len) // 2
        return seq[start:start + target_len]
    else:
        # Pad with X on both sides
        pad_total = target_len - len(seq)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return 'X' * pad_left + seq + 'X' * pad_right


def get_pfam_labels(item):
    """Extract Pfam family labels from dataset item.

    Returns a list of Pfam IDs like ['PF00005', 'PF00069'].

    Note: labels_str is stored as a STRING like "['Pfam:PF00118']"
    rather than an actual list, so we need to parse it.
    """
    import ast

    labels_str = item.get("labels_str")
    if labels_str is None:
        return []

    # Handle string representation of list
    if isinstance(labels_str, str):
        try:
            labels_str = ast.literal_eval(labels_str)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(labels_str, (list, tuple)):
        labels_str = [labels_str]

    # Extract just the PF##### part from 'Pfam:PF00005'
    result = []
    for label in labels_str:
        if isinstance(label, str):
            # Handle 'Pfam:PF00005' format
            if ":" in label:
                result.append(label.split(":")[-1])
            else:
                result.append(label)
    return result


def load_and_filter_dataset():
    """Load SwissProt-Pfam and filter to top families with appropriate lengths.

    Strategy: First filter by length, then pick top families from those
    that remain. This ensures we have sequences in both dimensions.
    """
    from datasets import load_dataset

    print("Loading DanielHesslow/SwissProt-Pfam...")
    ds = load_dataset("DanielHesslow/SwissProt-Pfam", split="train")

    # Debug: print first item structure
    first_item = ds[0]
    print(f"Dataset columns: {list(first_item.keys())}")
    print(f"Sample labels: {first_item.get('labels', 'N/A')}")
    print(f"Sample labels_str: {first_item.get('labels_str', 'N/A')}")
    print(f"Extracted: {get_pfam_labels(first_item)}")

    # First pass: filter by length and count families WITHIN length range
    length_counts = Counter()
    family_counts_in_range = Counter()
    items_in_range = []

    for item in ds:
        seq = item["seq"]
        seq_len = len(seq)
        length_counts[seq_len] += 1

        if MIN_LEN <= seq_len <= MAX_LEN:
            pfam_ids = get_pfam_labels(item)
            for pfam_id in pfam_ids:
                family_counts_in_range[pfam_id] += 1
            if pfam_ids:
                items_in_range.append((seq, pfam_ids))

    print(f"Sequence length distribution (top 10): {length_counts.most_common(10)}")
    print(f"Sequences in length range [{MIN_LEN}, {MAX_LEN}]: {len(items_in_range)}")
    print(f"Unique families in range: {len(family_counts_in_range)}")

    # Get top N families by count WITHIN length range
    top_families = [fam for fam, _ in family_counts_in_range.most_common(N_FAMILIES)]
    print(f"Top {N_FAMILIES} families (in length range): {top_families}")
    print(f"Counts: {[family_counts_in_range[f] for f in top_families]}")
    family_to_idx = {fam: i for i, fam in enumerate(top_families)}

    # Second pass: filter to sequences belonging to top families
    filtered_seqs = []
    filtered_labels = []

    for seq, pfam_ids in items_in_range:
        # Check if belongs to top families
        matching = [p for p in pfam_ids if p in family_to_idx]
        if matching:
            label = matching[0]  # Take first matching family
            filtered_seqs.append(seq)
            filtered_labels.append(family_to_idx[label])

    print(f"Filtered to {len(filtered_seqs)} sequences in top families")

    # Deduplicate
    seen = set()
    unique_seqs = []
    unique_labels = []
    for seq, label in zip(filtered_seqs, filtered_labels):
        if seq not in seen:
            seen.add(seq)
            unique_seqs.append(seq)
            unique_labels.append(label)

    print(f"After deduplication: {len(unique_seqs)} sequences")

    # Pad/crop to fixed length
    processed_seqs = [pad_or_crop(s) for s in unique_seqs]

    return processed_seqs, unique_labels, top_families


def encode_with_esm(sequences, batch_size=32, device="cuda"):
    """Encode sequences with frozen ESM-2, return per-residue hidden states."""
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading ESM-2 model (facebook/esm2_t6_8M_UR50D)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device).eval()

    all_hidden = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=SEQ_LEN + 2  # +2 for special tokens
            ).to(device)

            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, L+2, 320]

            # Remove BOS/EOS tokens, keep only residue positions
            # ESM adds <cls> at start and <eos> at end
            hidden = hidden[:, 1:SEQ_LEN + 1, :]  # [B, SEQ_LEN, 320]

            all_hidden.append(hidden.cpu())

            if (i // batch_size) % 10 == 0:
                print(f"  Encoded {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return torch.cat(all_hidden, dim=0)


def build_cache(out, seed=0, device="cuda", limit=None):
    """Build the cached dataset with ESM-2 embeddings."""
    sequences, labels, family_names = load_and_filter_dataset()

    if limit:
        sequences = sequences[:limit]
        labels = labels[:limit]

    print(f"Clustering {len(sequences)} sequences...")
    cluster_ids = greedy_cluster(sequences)
    n_clusters = len(set(cluster_ids))
    print(f"Found {n_clusters} clusters")

    print("Splitting by clusters...")
    train_idx, val_idx, test_idx = split_by_clusters(cluster_ids, seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print("Encoding with ESM-2...")
    z = encode_with_esm(sequences, device=device)  # [N, SEQ_LEN, 320]

    # Convert to tensors
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Compute standardization stats from training set only
    train_z = z[train_idx]
    z_mean = train_z.mean()
    z_std = train_z.std()
    print(f"Latent stats: mean={z_mean:.4f}, std={z_std:.4f}")

    # Standardize all data
    z = (z - z_mean) / z_std

    # Save
    cache = {
        "z": z,
        "labels": labels_t,
        "sequences": sequences,
        "family_names": family_names,
        "train_idx": torch.tensor(train_idx),
        "val_idx": torch.tensor(val_idx),
        "test_idx": torch.tensor(test_idx),
        "z_mean": z_mean,
        "z_std": z_std,
        "n_families": N_FAMILIES,
        "seq_len": SEQ_LEN,
        "hidden_dim": HIDDEN_DIM,
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"Saved cache to {out}")

    return cache


class PfamData:
    """Holds the three splits plus statistics for sampling."""

    def __init__(self, cache_path, limit=None, seed=0):
        d = torch.load(cache_path)

        self.z_mean = d["z_mean"]
        self.z_std = d["z_std"]
        self.n_families = d["n_families"]
        self.family_names = d["family_names"]
        self.seq_len = d["seq_len"]
        self.hidden_dim = d["hidden_dim"]
        self.sequences = d["sequences"]

        z = d["z"]
        labels = d["labels"]

        # Build splits
        self.splits = {}
        for split_name, idx_key in [("train", "train_idx"),
                                      ("val", "val_idx"),
                                      ("test", "test_idx")]:
            idx = d[idx_key]
            if limit and split_name == "train":
                idx = idx[:limit]
            self.splits[split_name] = (z[idx], labels[idx])

        # Family distribution from training set
        train_labels = self.splits["train"][1]
        self.family_hist = torch.bincount(train_labels, minlength=self.n_families).float()
        self.family_hist /= self.family_hist.sum()

        # Training sequences for novelty check
        train_idx = d["train_idx"].tolist()
        self.train_sequences = set(self.sequences[i] for i in train_idx)

    def loader(self, split, batch_size, shuffle=True):
        z, labels = self.splits[split]
        ts = torch.utils.data.TensorDataset(z, labels)
        return torch.utils.data.DataLoader(
            ts, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the Pfam-ESM2 tensor cache")
    p.add_argument("--out", default="data/pfam_esm2.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Limit sequences for debugging")
    a = p.parse_args()

    build_cache(a.out, a.seed, a.device, a.limit)
