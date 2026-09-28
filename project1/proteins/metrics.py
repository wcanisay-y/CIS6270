"""Evaluation metrics for generated protein sequences.

Metrics:
    family_accuracy   fraction whose predicted family matches requested
    uniqueness        distinct sequences / total valid sequences
    novelty           sequences absent from training set / unique sequences
    diversity         1 - mean pairwise sequence identity (higher = more diverse)

Sequences are decoded from ESM-2 latents via lm_head argmax.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from guidance import get_decoder
from sample import load


def decode_sequences(z, z_mean, z_std, device="cuda"):
    """Decode latent states to amino acid sequences."""
    decoder = get_decoder(device)
    return decoder.decode_to_sequence(z.to(device), z_mean, z_std)


def sequence_identity(seq1, seq2):
    """Compute sequence identity (fraction of matching residues)."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    if len(seq1) == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def mean_pairwise_identity(sequences, max_pairs=10000):
    """Compute mean pairwise sequence identity.

    For large sets, samples pairs randomly to avoid O(n^2) computation.
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        # Compute all pairs
        identities = []
        for i in range(n):
            for j in range(i + 1, n):
                identities.append(sequence_identity(sequences[i], sequences[j]))
    else:
        # Sample pairs
        rng = np.random.RandomState(0)
        identities = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, 2, replace=False)
            identities.append(sequence_identity(sequences[i], sequences[j]))

    return np.mean(identities)


def classify_family(z, classifier, device="cuda"):
    """Predict family from latent states using trained classifier."""
    with torch.no_grad():
        logits = classifier(z.to(device))
        return logits.argmax(dim=-1)


def evaluate(samples, train_sequences=None, classifier=None, device="cuda"):
    """Evaluate generated samples.

    Args:
        samples: dict with keys z, family, z_mean, z_std
        train_sequences: set of training sequences for novelty
        classifier: trained FamilyClassifier for accuracy

    Returns:
        dict of metrics
    """
    z = samples["z"].to(device)
    target_family = samples["family"].to(device)
    z_mean = float(samples["z_mean"])
    z_std = float(samples["z_std"])

    # Decode to sequences
    print("Decoding sequences...")
    sequences = decode_sequences(z, z_mean, z_std, device)
    n_samples = len(sequences)

    # Filter out sequences that are mostly padding (X)
    valid_sequences = [s for s in sequences if s.count('X') < len(s) * 0.5]
    n_valid = len(valid_sequences)

    results = {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "valid_rate": n_valid / max(n_samples, 1),
    }

    # Uniqueness
    unique_sequences = set(valid_sequences)
    results["uniqueness"] = len(unique_sequences) / max(n_valid, 1)

    # Novelty (if training set provided)
    if train_sequences is not None:
        novel = [s for s in unique_sequences if s not in train_sequences]
        results["novelty"] = len(novel) / max(len(unique_sequences), 1)

    # Diversity (1 - mean identity)
    if len(valid_sequences) >= 2:
        print("Computing diversity...")
        mean_id = mean_pairwise_identity(valid_sequences)
        results["diversity"] = 1.0 - mean_id
        results["mean_pairwise_identity"] = mean_id

    # Family accuracy (if classifier provided)
    if classifier is not None:
        print("Computing family accuracy...")
        pred_family = classify_family(z, classifier, device)
        correct = (pred_family == target_family).float()
        results["family_accuracy"] = correct.mean().item()

    # Add metadata
    results["nfe"] = samples.get("nfe")
    results["mean_gate"] = samples.get("mean_gate")

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, help="path to generated samples .pt")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--data", default="data/pfam_esm2.pt", help="for novelty check")
    p.add_argument("--classifier-ckpt", default=None, help="trained FamilyClassifier")
    p.add_argument("--no-novelty", action="store_true")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.load(a.samples, weights_only=False)

    # Load training sequences for novelty
    train_sequences = None
    if not a.no_novelty:
        try:
            from data.pfam import PfamData
            data = PfamData(a.data)
            train_sequences = data.train_sequences
            print(f"Loaded {len(train_sequences)} training sequences for novelty check")
        except Exception as e:
            print(f"Warning: could not load training data for novelty: {e}")

    # Load classifier
    classifier = None
    if a.classifier_ckpt:
        classifier, _ = load(a.classifier_ckpt, device)
        print(f"Loaded classifier from {a.classifier_ckpt}")

    # Evaluate
    results = evaluate(samples, train_sequences, classifier, device)
    results["config"] = samples.get("config")

    print("\n" + "=" * 50)
    print("Results:")
    print(json.dumps(results, indent=2))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()
