import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_FILE = "/home/lunsusa/dtu/algorithms/project/TMHMM/data/train_test_data/hobohm1_ole_formula/downsized/train_dataset_fused_&_downsized_80.0%.txt"
TEST_FILE  = "/home/lunsusa/dtu/algorithms/project/TMHMM/data/train_test_data/hobohm1_ole_formula/downsized/test_dataset_fused_&_downsized20.0%.txt"
OUT_DIR    = "/home/lunsusa/dtu/algorithms/project/TMHMM/results/data_summary/downsized"

# Map the tag in the header to a display name
CLASS_DISPLAY = {
    "glob":   "GLOBULAR",
    "signal": "SIGNAL",
    "sp_tm":  "SP+TM",
    "tm":     "TM",
}

STATE_NAME  = {"S": "Signal peptide", "I": "Inside", "M": "Membrane", "O": "Outside"}
STATE_ORDER = ["S", "I", "M", "O"]
COLORS_CLASS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
COLORS_STATE = {"S": "#DD8452", "I": "#4C72B0", "M": "#C44E52", "O": "#55A868"}




def load_fasta(filepath):
    """
    Reads a fused fasta file where the class tag is in the header:
        >PROTEINID|class_tag
        SEQUENCE
        TOPOLOGY

    Returns a list of dicts with keys: header, class_tag, seq, top.
    """
    records = []

    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            header    = lines[i][1:]                        # e.g. "O60939|sp_tm"
            class_tag = header.split("|")[-1]               # e.g. "sp_tm"
            seq       = lines[i + 1]
            top       = lines[i + 2]
            records.append({
                "header":    header,
                "class_tag": class_tag,
                "seq":       seq,
                "top":       top,
            })
            i += 3
        else:
            i += 1

    return records


def compute_stats(records, split_label):
    """
    Given a list of records, groups by class_tag and computes per-class
    and overall sequence/residue/state statistics.

    Returns
    -------
    rows          : list of dicts (one per class + one overall)
    all_lengths   : dict  class_display_name -> np.ndarray of lengths
    all_state_counts : dict  class_display_name -> {state: count}
    """
    # Group by class
    groups = {}
    for rec in records:
        tag = rec["class_tag"]
        if tag not in CLASS_DISPLAY:
            continue
        name = CLASS_DISPLAY[tag]
        groups.setdefault(name, []).append(rec)

    rows             = []
    all_lengths      = {}
    all_state_counts = {}
    combined_seqs    = []
    combined_tops    = []

    for name in CLASS_DISPLAY.values():      # fixed display order
        if name not in groups:
            continue
        recs    = groups[name]
        seqs    = [r["seq"] for r in recs]
        tops    = [r["top"] for r in recs]
        lengths = np.array([len(s) for s in seqs])

        state_counts = {s: 0 for s in STATE_ORDER}
        for top in tops:
            for ch in top:
                if ch in state_counts:
                    state_counts[ch] += 1
        total_res = sum(state_counts.values())

        row = {
            "split":        split_label,
            "class":        name,
            "n_sequences":  len(seqs),
            "n_residues":   total_res,
            "length_min":   int(lengths.min()),
            "length_mean":  round(lengths.mean(), 1),
            "length_median":int(np.median(lengths)),
            "length_max":   int(lengths.max()),
        }
        for s in STATE_ORDER:
            row[f"pct_{s}"] = round(100 * state_counts[s] / total_res, 2) if total_res else 0.0

        rows.append(row)
        all_lengths[name]      = lengths
        all_state_counts[name] = state_counts
        combined_seqs.extend(seqs)
        combined_tops.extend(tops)

    # Overall row
    lengths_all   = np.concatenate(list(all_lengths.values()))
    state_all     = {s: sum(all_state_counts[n][s] for n in all_state_counts) for s in STATE_ORDER}
    total_res_all = sum(state_all.values())
    overall = {
        "split":        split_label,
        "class":        "ALL",
        "n_sequences":  sum(r["n_sequences"] for r in rows),
        "n_residues":   total_res_all,
        "length_min":   int(lengths_all.min()),
        "length_mean":  round(lengths_all.mean(), 1),
        "length_median":int(np.median(lengths_all)),
        "length_max":   int(lengths_all.max()),
    }
    for s in STATE_ORDER:
        overall[f"pct_{s}"] = round(100 * state_all[s] / total_res_all, 2) if total_res_all else 0.0

    rows.append(overall)
    return rows, all_lengths, all_state_counts



def class_names_present(all_lengths):
    return [n for n in CLASS_DISPLAY.values() if n in all_lengths]


def fig_class_counts(all_lengths_train, all_lengths_test, out_path):
    names  = class_names_present(all_lengths_train)
    x      = np.arange(len(names))
    width  = 0.35
    counts_train = [len(all_lengths_train[n]) for n in names]
    counts_test  = [len(all_lengths_test.get(n, [])) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_tr = ax.bar(x - width/2, counts_train, width, label="Train", color="#4C72B0")
    bars_te = ax.bar(x + width/2, counts_test,  width, label="Test",  color="#4C72B0", alpha=0.5)

    for b, c in zip(bars_tr, counts_train):
        ax.text(b.get_x() + b.get_width()/2, c, str(c), ha="center", va="bottom", fontsize=8)
    for b, c in zip(bars_te, counts_test):
        ax.text(b.get_x() + b.get_width()/2, c, str(c), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of sequences")
    ax.set_title("Sequences per topology class — train vs test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_state_composition(all_state_counts, title, out_path):
    names = class_names_present(all_state_counts)
    x     = np.arange(len(names))
    bottom = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in STATE_ORDER:
        vals = np.array([
            100 * all_state_counts[n][s] / sum(all_state_counts[n].values())
            for n in names
        ])
        ax.bar(x, vals, bottom=bottom, label=f"{s} ({STATE_NAME[s]})", color=COLORS_STATE[s])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% of residues")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_length_distribution(all_lengths, title, out_path):
    names  = class_names_present(all_lengths)
    colors = {n: c for n, c in zip(CLASS_DISPLAY.values(), COLORS_CLASS)}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        [all_lengths[n] for n in names],
        bins=30, stacked=True,
        label=names,
        color=[colors[n] for n in names],
    )
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_counts_colored(all_lengths, all_state_counts, title, out_path):
    """Bar chart: bar height = n_sequences, bar fill = residue state composition."""
    names  = class_names_present(all_lengths)
    x      = np.arange(len(names))
    counts = np.array([len(all_lengths[n]) for n in names])
    bottom = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in STATE_ORDER:
        pct = np.array([
            100 * all_state_counts[n][s] / sum(all_state_counts[n].values())
            for n in names
        ])
        heights = counts * pct / 100.0
        ax.bar(x, heights, bottom=bottom, label=f"{s} ({STATE_NAME[s]})", color=COLORS_STATE[s])
        bottom += heights

    for xi, c in zip(x, counts):
        ax.text(xi, c, str(c), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of sequences")
    ax.set_title(title, pad=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_records = load_fasta(TRAIN_FILE)
    test_records  = load_fasta(TEST_FILE)
    all_records   = train_records + test_records

    print(f"Total records: {len(all_records)}")

    rows, all_lengths, all_state_counts = compute_stats(all_records, "all")

    # Save stats CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "data_summary_stats.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # Figure 1: sequence counts per class
    names  = class_names_present(all_lengths)
    x      = np.arange(len(names))
    counts = [len(all_lengths[n]) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, counts, color=COLORS_CLASS[:len(names)])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("Number of sequences")
    ax.set_title("Sequences per topology class")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_class_counts.png"), dpi=200)
    plt.close(fig)

    # Figure 2: state composition
    fig_state_composition(
        all_state_counts,
        "Per-residue topology-state composition",
        os.path.join(OUT_DIR, "fig_state_composition.png"),
    )

    # Figure 3: sequence length distribution
    fig_length_distribution(
        all_lengths,
        "Sequence length distribution",
        os.path.join(OUT_DIR, "fig_lengths.png"),
    )

    # Figure 4: counts colored by residue state
    fig_counts_colored(
        all_lengths, all_state_counts,
        "Sequences per class, colored by residue-state composition",
        os.path.join(OUT_DIR, "fig_counts_colored.png"),
    )

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()