from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from librosa.sequence import dtw


output_dir = Path(
    "/scratch5/fazh9208/clarity/recipes/whisper_si/results"
)

metadata = pd.read_csv(output_dir / "metadata.csv")


def mean_cosine(x, y):
    """Mean frame-wise cosine similarity."""
    x = x.squeeze(0).numpy()
    y = y.squeeze(0).numpy()

    n = min(len(x), len(y))

    return np.mean([
        1 - cosine(x[i], y[i])
        for i in range(n)
    ])


def dtw_cosine(x, y):
    """DTW (dynamic time warping) -aligned cosine similarity."""
    if x is None or y is None:
        return np.nan

    x = x.squeeze(0).numpy()
    y = y.squeeze(0).numpy()

    distance = cdist(x, y, metric="cosine")

    _, path = dtw(C=distance)

    return np.mean([
        1 - distance[i, j]
        for i, j in path
    ])


rows = []

for _, item in metadata.iterrows():

    print(item["signal"])

    data = torch.load(
        item["hidden_file"],
        map_location="cpu",
        weights_only=False,
    )

    row = {
        "signal": item["signal"],
        "human_si": item["human_correctness"],
    }

    # Encoder: 24 layers
    for i, (clean, enhanced) in enumerate(zip(
        data["encoder_clean"],
        data["encoder_enhanced"],
    )):
        row[f"enc_{i + 1:02d}"] = mean_cosine(
            clean,
            enhanced,
        )

    # Decoder: 24 layers
    dec_clean = data["decoder_clean"]
    dec_enhanced = data["decoder_enhanced"]

    for i in range(24):

        if dec_clean is None or dec_enhanced is None:
            score = np.nan
        else:
            score = dtw_cosine(
                dec_clean[i],
                dec_enhanced[i],
            )

        row[f"dec_{i + 1:02d}"] = score

    rows.append(row)


results = pd.DataFrame(rows)

results.to_csv(
    output_dir / "raw_scores.csv",
    index=False,
)

print(results.head())

