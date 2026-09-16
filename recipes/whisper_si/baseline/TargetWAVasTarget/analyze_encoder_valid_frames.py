from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


# Paths
results_dir = Path(
    "/scratch5/fazh9208/clarity/"
    "recipes/whisper_si/results"
)

analysis_dir = results_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

metadata = pd.read_csv(results_dir / "metadata.csv")



# Valid-frame encoder similarity
def valid_encoder_frames(num_samples):
    """Whisper encoder: approximately 320 audio samples per frame."""
    return int(np.ceil(num_samples / 320))


def mean_cosine_valid(clean, enhanced, n_clean, n_enhanced):

    clean = clean.squeeze(0).numpy()
    enhanced = enhanced.squeeze(0).numpy()

    # Only compare frames available in both signals
    n = min(
        n_clean,
        n_enhanced,
        len(clean),
        len(enhanced),
    )

    return np.mean([
        1 - cosine(clean[i], enhanced[i])
        for i in range(n)
    ])


# ============================================================
# Calculate 24 encoder layers
# ============================================================

rows = []

for index, item in metadata.iterrows():

    data = torch.load(
        item["hidden_file"],
        map_location="cpu",
        weights_only=False,
    )

    n_clean = valid_encoder_frames(
        data["num_samples_clean"]
    )

    n_enhanced = valid_encoder_frames(
        data["num_samples_enhanced"]
    )

    row = {
        "signal": item["signal"],
        "human_si": item["human_correctness"],
        "n_frames_clean": n_clean,
        "n_frames_enhanced": n_enhanced,
    }

    for layer, (clean, enhanced) in enumerate(
        zip(
            data["encoder_clean"],
            data["encoder_enhanced"],
        ),
        start=1,
    ):

        row[f"enc_valid_{layer:02d}"] = mean_cosine_valid(
            clean,
            enhanced,
            n_clean,
            n_enhanced,
        )

    rows.append(row)

    if (index + 1) % 500 == 0:
        print(f"{index + 1}/{len(metadata)}")


df = pd.DataFrame(rows)


# ============================================================
# Layer statistics
# ============================================================

stats = []

for layer in range(1, 25):

    col = f"enc_valid_{layer:02d}"

    stats.append({
        "layer": layer,
        "r": df[col].corr(df["human_si"]),
        "sd": df[col].std(),
    })

stats = pd.DataFrame(stats)

print("\nValid-frame encoder:")
print(stats.to_string(index=False))


# ============================================================
# Scatter plots
# ============================================================

fig, axes = plt.subplots(
    6,
    4,
    figsize=(14, 18),
    sharex=True,
    sharey=True,
)

for layer, ax in enumerate(axes.flatten(), start=1):

    col = f"enc_valid_{layer:02d}"
    r = df[col].corr(df["human_si"])

    ax.scatter(
        df[col],
        df["human_si"],
        s=4,
        alpha=0.25,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)

    ax.set_title(
        f"enc_{layer:02d}  r={r:.3f}"
    )

fig.supxlabel("Valid-frame cosine similarity")
fig.supylabel("Human SI (%)")

plt.tight_layout()

plt.savefig(
    analysis_dir / "encoder_valid_scatter.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Save
# ============================================================

df.to_csv(
    analysis_dir / "encoder_valid_scores.csv",
    index=False,
)

stats.to_csv(
    analysis_dir / "encoder_valid_statistics.csv",
    index=False,
)

print("\nSaved to:", analysis_dir)

