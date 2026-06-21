

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = "../../data/raw_data"
OUT_DIR = "../../results/data_summary"

# Classes 
CLASS_FILES = {
    "GLOBULAR": "GLOBULAR_sequences.3line",
    "SIGNAL":   "SIGNAL_sequences.3line",
    "SP+TM":    "SP+TM_sequences.3line",
    "TM":       "TM_sequences.3line",
}

STATE_NAME = {"S": "Signal peptide", "I": "Inside", "M": "Membrane", "O": "Outside"}
STATE_ORDER = ["S", "I", "M", "O"]


# Load

def load_3line(filepath):
    
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    ids, seqs, tops = [], [], []
    for i in range(0, len(lines), 3):
        header, seq, top = lines[i], lines[i + 1], lines[i + 2]
        protein_id = header.lstrip(">").split("|")[0]
        ids.append(protein_id)
        seqs.append(seq)
        tops.append(top)
    return ids, seqs, tops


# Stats

def class_stats(name, seqs, tops):
    lengths = np.array([len(s) for s in seqs])

    state_counts = {s: 0 for s in STATE_ORDER}
    for top in tops:
        for ch in top:
            if ch in state_counts:
                state_counts[ch] += 1
    total_residues = sum(state_counts.values())

    row = {
        "class": name,
        "n_sequences": len(seqs),
        "n_residues": total_residues,
        "length_min": lengths.min(),
        "length_mean": round(lengths.mean(), 1),
        "length_median": int(np.median(lengths)),
        "length_max": lengths.max(),
    }
    for s in STATE_ORDER:
        row[f"pct_{s}"] = round(100 * state_counts[s] / total_residues, 2) if total_residues else 0.0

    return row, lengths, state_counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    all_lengths = {}
    all_state_counts = {}
    combined_seqs, combined_tops, combined_aas = [], [], []

    for name, fname in CLASS_FILES.items():
        filepath = os.path.join(RAW_DIR, fname)
        ids, seqs, tops = load_3line(filepath)
        row, lengths, state_counts = class_stats(name, seqs, tops)
        all_rows.append(row)
        all_lengths[name] = lengths
        all_state_counts[name] = state_counts
        combined_seqs.extend(seqs)
        combined_tops.extend(tops)
        combined_aas.extend(list("".join(seqs)))

    # Combined 
    combined_row, combined_lengths, combined_state_counts = class_stats(
        "ALL (excl. BETA)", combined_seqs, combined_tops
    )
    all_rows.append(combined_row)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "data_summary_stats.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # Amino-acid composition (global, excl. BETA)
    aa_counts = pd.Series(combined_aas).value_counts().sort_index()
    aa_pct = (100 * aa_counts / aa_counts.sum()).round(2)

    
    # Figure 1: number of sequences per class
    
    plt.figure(figsize=(6, 4))
    counts = [r["n_sequences"] for r in all_rows[:-1]]
    names = [r["class"] for r in all_rows[:-1]]
    bars = plt.bar(names, counts, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
    plt.ylabel("Number of sequences")
    plt.title("Raw data: sequences per topology class (BETA excluded)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_class_counts.png"), dpi=200)
    plt.close()

    
    # Figure 2: per-residue state composition, stacked bar per class
    
    plt.figure(figsize=(7, 4))
    bottom = np.zeros(len(CLASS_FILES))
    x = np.arange(len(CLASS_FILES))
    colors = {"S": "#DD8452", "I": "#4C72B0", "M": "#C44E52", "O": "#55A868"}
    for s in STATE_ORDER:
        vals = np.array([
            100 * all_state_counts[name][s] / sum(all_state_counts[name].values())
            for name in CLASS_FILES
        ])
        plt.bar(x, vals, bottom=bottom, label=f"{s} ({STATE_NAME[s]})", color=colors[s])
        bottom += vals
    plt.xticks(x, list(CLASS_FILES.keys()))
    plt.ylabel("% of residues")
    plt.title("Per-residue topology-state composition by class")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_state_composition.png"), dpi=200)
    plt.close()


    # Figure 3: number of sequences per class, colored by per-residue state composition

    plt.figure(figsize=(7, 5))
    x = np.arange(len(CLASS_FILES))
    counts = np.array([r["n_sequences"] for r in all_rows[:-1]])
    names = [r["class"] for r in all_rows[:-1]]

    colors = {"S": "#DD8452", "I": "#4C72B0", "M": "#C44E52", "O": "#55A868"}
    bottom = np.zeros(len(CLASS_FILES))

    for s in STATE_ORDER:
        pct = np.array([
            100 * all_state_counts[name][s] / sum(all_state_counts[name].values())
            for name in CLASS_FILES
        ])
        heights = counts * pct / 100.0
        plt.bar(
            x,
            heights,
            bottom=bottom,
            label=f"{s} ({STATE_NAME[s]})",
            color=colors[s],
        )
        bottom += heights

    for xi, c in zip(x, counts):
        plt.text(xi, c, str(c), ha="center", va="bottom")

    plt.xticks(x, names)
    plt.ylabel("Number of sequences")
    plt.title("Sequences per topology class, colored by residue-state composition", pad=15)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_class_counts2.png"), dpi=200)
    plt.close()


    
    # Figure 4: sequence length distribution
    
    plt.figure(figsize=(6, 4))
    plt.hist(
        [all_lengths[n] for n in CLASS_FILES],
        bins=30, stacked=True, label=list(CLASS_FILES.keys()),
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    )
    plt.xlabel("Sequence length (residues)")
    plt.ylabel("Count")
    plt.title("Sequence length distribution (raw data, BETA excluded)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_sequence_lengths.png"), dpi=200)
    plt.close()

    
    

if __name__ == "__main__":
    main()