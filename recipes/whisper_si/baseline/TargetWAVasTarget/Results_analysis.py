from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

results_dir = Path(
    "/scratch5/fazh9208/clarity/"
    "recipes/whisper_si/results"
)

analysis_dir = results_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

scores = pd.read_csv(results_dir / "raw_scores.csv")
metadata = pd.read_csv(results_dir / "metadata.csv")


# ============================================================
# WCS
# ============================================================

def normalize_text(text):
    """Lowercase, remove punctuation, split into words."""

    if pd.isna(text):
        return []

    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)

    return text.split()


def word_correctness(reference, hypothesis):

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if len(ref) == 0:
        return np.nan, ""

    hyp_count = Counter(hyp)
    correct_words = []

    # Follow reference order
    for word in ref:
        if hyp_count[word] > 0:
            correct_words.append(word)
            hyp_count[word] -= 1

    wcs = 100 * len(correct_words) / len(ref)

    return wcs, " ".join(correct_words)


wcs_results = [
    word_correctness(prompt, transcript)
    for prompt, transcript in zip(
        metadata["prompt"],
        metadata["transcript_enhanced"],
    )
]

wcs = [x[0] for x in wcs_results]
correct_words = [x[1] for x in wcs_results]

# Insert after transcript_enhanced
pos = metadata.columns.get_loc("transcript_enhanced") + 1

metadata.insert(
    pos,
    "wcs",
    wcs,
)

metadata.insert(
    pos + 1,
    "correct_words",
    correct_words,
)

# Save metadata with WCS results
metadata.to_csv(
    analysis_dir / "metadata_with_wcs.csv",
    index=False,
)

print(
    "Saved:",
    analysis_dir / "metadata_with_wcs.csv"
)



df = scores.merge(
    metadata[["signal", "wcs"]],
    on="signal",
    how="left",
)



# Layer statistics
def get_layer_statistics(df, prefix):

    rows = []

    for layer in range(1, 25):

        col = f"{prefix}_{layer:02d}"

        valid = df[
            [col, "human_si"]
        ].dropna()

        rows.append({
            "layer": layer,
            "r": valid[col].corr(valid["human_si"]),
            "sd": valid[col].std(),
            "n": len(valid),
        })

    return pd.DataFrame(rows)


enc_stats = get_layer_statistics(df, "enc")
dec_stats = get_layer_statistics(df, "dec")


print("\nEncoder:")
print(enc_stats.to_string(index=False))

print("\nDecoder:")
print(dec_stats.to_string(index=False))



# 24-layer scatter plots
def plot_layer_scatter(df, prefix, title):

    fig, axes = plt.subplots(
        6,
        4,
        figsize=(14, 18),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for layer, ax in enumerate(axes, start=1):

        col = f"{prefix}_{layer:02d}"

        valid = df[
            [col, "human_si"]
        ].dropna()

        r = valid[col].corr(
            valid["human_si"]
        )

        ax.scatter(
            valid[col],
            valid["human_si"],
            s=4,
            alpha=0.25,
        )

        # Fixed axes for direct comparison across layers
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)

        ax.set_title(
            f"{prefix}_{layer:02d}  r={r:.3f}"
        )

    fig.supxlabel("Cosine similarity")
    fig.supylabel("Human SI (%)")
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    plt.savefig(
        analysis_dir / f"{prefix}_scatter.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


plot_layer_scatter(
    df,
    "enc",
    "Whisper Encoder Hidden Representations",
)

plot_layer_scatter(
    df,
    "dec",
    "Whisper Decoder Hidden Representations",
)



# Correlation and SD across layers
def plot_layer_statistics(stats, name):

    fig, ax1 = plt.subplots(
        figsize=(10, 5)
    )

    ax1.plot(
        stats["layer"],
        stats["r"],
        marker="o",
    )

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Pearson r")
    ax1.set_xticks(range(1, 25))
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        stats["layer"],
        stats["sd"],
        marker="s",
        linestyle="--",
    )

    ax2.set_ylabel(
        "SD of cosine similarity"
    )

    plt.title(
        f"{name}: correlation and similarity variation"
    )

    plt.tight_layout()

    plt.savefig(
        analysis_dir /
        f"{name.lower()}_layer_statistics.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


plot_layer_statistics(
    enc_stats,
    "Encoder",
)

plot_layer_statistics(
    dec_stats,
    "Decoder",
)


# WCS vs Human SI
valid_wcs = df[
    ["wcs", "human_si"]
].dropna()

wcs_r = valid_wcs["wcs"].corr(
    valid_wcs["human_si"]
)


print("\nWCS:")
print(f"Pearson r = {wcs_r:.3f}")
print(f"N = {len(valid_wcs)}")
print(f"Mean WCS = {valid_wcs['wcs'].mean():.2f}")
print(f"SD WCS = {valid_wcs['wcs'].std():.2f}")


plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    valid_wcs["wcs"],
    valid_wcs["human_si"],
    s=5,
    alpha=0.25,
)

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xlabel(
    "Whisper word correctness (%)"
)

plt.ylabel(
    "Human SI (%)"
)

plt.title(
    f"Whisper WCS vs Human SI  "
    f"r={wcs_r:.3f}"
)

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    analysis_dir /
    "wcs_vs_human_si.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Save numerical results
# ============================================================

enc_stats.to_csv(
    analysis_dir /
    "encoder_layer_statistics.csv",
    index=False,
)

dec_stats.to_csv(
    analysis_dir /
    "decoder_layer_statistics.csv",
    index=False,
)

df[
    ["signal", "human_si", "wcs"]
].to_csv(
    analysis_dir /
    "wcs_scores.csv",
    index=False,
)


print("\nSaved analysis to:")
print(analysis_dir)