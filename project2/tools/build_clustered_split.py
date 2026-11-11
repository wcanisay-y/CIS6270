"""Build a homology-purged training corpus for the clustered-split retrain.

The leakage question this answers
--------------------------------
`G_clean_conditioning_note.md` reports chemical-minus-random conditioning at
+20.22 pLDDT on 96 pattern sources whose maximum 10-mer containment against
training is 0.026. 10-mer containment is a substring test, not an alignment,
so it controls memorised fragments and NOT remote homology: a source can
share a fold with hundreds of training sequences while sharing no 10-mer.

Design
------
Rather than select pattern sources that happen to be orphans -- which would
bias the set towards unusual proteins with no SwissProt relatives -- this
keeps a representative sample of sources and removes their relatives from
TRAINING:

  select : sample N held-out pattern sources H from the dev candidates,
           with no homology criterion, and write them as FASTA
  (BLAST)  H vs training
  expand : optional, depth 2 -- write the round-1 hits as FASTA
  (BLAST)  those hits vs training
  finish : train_purged.txt = train minus every relative found

The retrained model has then never seen a relative of any pattern source.
Conditioning on H is measured on BOTH the original and the purged
checkpoints with identical patterns, so the difference between the two is
exactly the contribution of having homologs in training.
"""
import argparse
import collections
import json
import os

import numpy as np


def read_hits(path, max_evalue):
    """BLAST outfmt 6 (qseqid sseqid pident evalue ...) -> {query: {subject}}."""
    q2s = collections.defaultdict(set)
    n = kept = 0
    if not os.path.exists(path):
        raise SystemExit("missing BLAST output %s" % path)
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            n += 1
            if float(f[3]) <= max_evalue:
                q2s[f[0]].add(f[1])
                kept += 1
    print("  %s: %d HSP rows, %d kept at E <= %g, %d queries with a hit"
          % (os.path.basename(path), n, kept, max_evalue, len(q2s)))
    return q2s


def write_fasta(path, pairs):
    with open(path, "w") as fh:
        for name, seq in pairs:
            fh.write(">%s\n%s\n" % (name, seq))
    return len(pairs)


def stage_select(a):
    cand = open(os.path.join(a.dir, "devcand.txt")).read().split("\n")
    rng = np.random.default_rng(a.seed)
    sel = sorted(rng.choice(len(cand), min(a.n_patterns, len(cand)),
                            replace=False).tolist())
    pairs = [("cd%d" % i, cand[i]) for i in sel]
    write_fasta(os.path.join(a.dir, "h.fasta"), pairs)
    json.dump(dict(dev_indices=sel, seed=a.seed, n_candidates=len(cand)),
              open(os.path.join(a.dir, "heldout_sel.json"), "w"), indent=1)
    print("selected %d of %d dev candidates (seed %d) -> h.fasta"
          % (len(sel), len(cand), a.seed))


def stage_expand(a):
    train = open(os.path.join(a.dir, "train.txt")).read().split("\n")
    sel = json.load(open(os.path.join(a.dir, "heldout_sel.json")))
    keep_q = {"cd%d" % i for i in sel["dev_indices"]}
    r1 = read_hits(os.path.join(a.dir, "h_vs_train.tsv"), a.evalue)
    direct = set()
    for q, subs in r1.items():
        if q in keep_q:
            direct |= subs
    pairs = [(s, train[int(s[2:])]) for s in sorted(direct)
             if s.startswith("tr") and int(s[2:]) < len(train)]
    write_fasta(os.path.join(a.dir, "purge.fasta"), pairs)
    print("round-1 relatives: %d -> purge.fasta (for depth-2 expansion)"
          % len(pairs))


def stage_finish(a):
    D = a.dir
    train = open(os.path.join(D, "train.txt")).read().split("\n")
    cand = open(os.path.join(D, "devcand.txt")).read().split("\n")
    sel = json.load(open(os.path.join(D, "heldout_sel.json")))["dev_indices"]
    H = [cand[i] for i in sel]
    print("corpus: %d training, %d dev candidates, %d held-out sources"
          % (len(train), len(cand), len(H)))

    print("\nround 1: pattern sources vs training")
    r1 = read_hits(os.path.join(D, "h_vs_train.tsv"), a.evalue)
    keep_q = {"cd%d" % i for i in sel}
    direct, with_rel = set(), 0
    for q, subs in r1.items():
        if q in keep_q:
            direct |= subs
            with_rel += 1
    print("  %d of %d sources have >= 1 relative; %d distinct training "
          "sequences are relatives" % (with_rel, len(H), len(direct)))

    purge = set(direct)
    added = 0
    if a.depth == 2:
        print("\nround 2: single-linkage expansion")
        r2 = read_hits(os.path.join(D, "purge_vs_train.tsv"), a.evalue)
        exp = set()
        for q, subs in r2.items():
            if q in direct:
                exp |= subs
        before = len(purge)
        purge |= exp
        added = len(purge) - before
        print("  expansion added %d (%d -> %d)" % (added, before, len(purge)))

    idx = {int(s[2:]) for s in purge
           if s.startswith("tr") and int(s[2:]) < len(train)}
    frac = len(idx) / max(len(train), 1)
    print("\npurge set: %d sequences = %.2f %% of the corpus"
          % (len(idx), 100 * frac))
    if frac > a.max_purge_frac:
        raise SystemExit(
            "REFUSING: a purge of %.1f %% would confound the comparison with "
            "corpus size. Re-run with a stricter --evalue or --depth 1."
            % (100 * frac))

    purged = [s for i, s in enumerate(train) if i not in idx]
    with open(os.path.join(D, "train_purged.txt"), "w") as fh:
        fh.write("\n".join(purged))

    ln = np.array([len(s) for s in H], dtype=float)
    la = np.array([len(s) for s in cand], dtype=float)
    out = dict(config=vars(a),
               n_train_before=len(train), n_train_after=len(purged),
               n_purged=len(idx), purge_frac=frac, depth2_added=added,
               n_patterns=len(H), n_patterns_with_relative=with_rel,
               n_direct_relatives=len(direct),
               heldout_len_mean=float(ln.mean()),
               candidate_len_mean=float(la.mean()),
               dev_indices=sel, seqs=H)
    json.dump(out, open(os.path.join(D, "heldout_patterns.json"), "w"),
              indent=1)

    print("purged corpus: %d sequences (was %d)" % (len(purged), len(train)))
    print("wrote train_purged.txt and heldout_patterns.json")
    print("\nheld-out length mean %.1f vs candidate pool %.1f" % (ln.mean(),
                                                                  la.mean()))
    print("fraction of sources WITH a training relative: %.3f -- a high "
          "value means the set is representative and the purge does real work"
          % (with_rel / max(len(H), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "expand", "finish"])
    ap.add_argument("--dir", default="data/clustered")
    ap.add_argument("--n-patterns", type=int, default=128)
    ap.add_argument("--evalue", type=float, default=1e-3,
                    help="a hit at or below this E-value counts as a relative")
    ap.add_argument("--depth", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max-purge-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"select": stage_select, "expand": stage_expand,
     "finish": stage_finish}[a.stage](a)


if __name__ == "__main__":
    main()
