"""Colab-ready MovieLens semantic ID training demo.

This script mirrors the toy semantic ID pipeline from the README but runs it on the
MovieLens "latest-small" dataset (~100k ratings, ~9.7k movies).  It is designed so you
can drop the entire file into a single Google Colab cell and execute it end-to-end.  The
code downloads MovieLens, learns semantic IDs with hierarchical K-Means, converts user
histories into those IDs, trains a compact Transformer decoder, and finally prints
recommendations for a held-out user sequence.

Usage (one Colab cell):

```
%pip -q install pandas==2.2.2 sentence-transformers==3.0.1 scikit-learn==1.5.1 \
    torch==2.3.1 datasets==2.20.0 tqdm==4.66.4

%run movielens_semantic_ids_demo.py
```

If you prefer running inside this repository clone, execute:

```
python notebooks/movielens_semantic_ids_demo.py
```

The script is intentionally verbose with comments so you can adapt or document it for
blog posts / LinkedIn recaps.
"""

from __future__ import annotations

import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 1. Reproducibility helpers
# ---------------------------------------------------------------------------

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Ensure deterministic runs across random, numpy, and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()


# ---------------------------------------------------------------------------
# 2. Download and prepare MovieLens metadata
# ---------------------------------------------------------------------------

DATA_ROOT = Path("./data/movielens-latest-small")
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(destination: Path = DATA_ROOT) -> Path:
    """Fetch and extract MovieLens latest-small if it is not already cached."""

    if destination.exists():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.with_suffix(".zip")
    print("Downloading MovieLens latest-small (~1 file, 9MB)...")
    torch.hub.download_url_to_file(DATASET_URL, archive_path)

    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    return destination / "ml-latest-small"


def load_catalog(dataset_dir: Path) -> pd.DataFrame:
    """Create a dataframe with movieId, title, genres, and semantic text."""

    movies = pd.read_csv(dataset_dir / "movies.csv")
    # Combine title and genres into a short textual summary for embedding.
    movies["text"] = movies.apply(
        lambda row: f"{row['title']} Genres: {row['genres'].replace('|', ', ')}.", axis=1
    )
    return movies


def load_interactions(dataset_dir: Path, min_rating: float = 3.5) -> pd.DataFrame:
    """Load ratings and keep only positive interactions above `min_rating`."""

    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    positive = ratings[ratings["rating"] >= min_rating].copy()
    positive.sort_values(["userId", "timestamp"], inplace=True)
    return positive


dataset_path = download_movielens()
catalog_df = load_catalog(dataset_path)
ratings_df = load_interactions(dataset_path)

print(f"Movies: {len(catalog_df):,} | Positive interactions: {len(ratings_df):,}")


# ---------------------------------------------------------------------------
# 3. Encode item metadata with a sentence transformer
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

movie_texts = catalog_df["text"].tolist()
with torch.inference_mode():
    movie_embeddings = encoder.encode(movie_texts, convert_to_tensor=True, device=device).cpu()


# ---------------------------------------------------------------------------
# 4. Learn hierarchical semantic IDs via residual K-Means
# ---------------------------------------------------------------------------

def train_hierarchical_kmeans(vectors: torch.Tensor, levels: Sequence[int]) -> List[KMeans]:
    """Fit a series of KMeans quantizers, subtracting residuals at each layer."""

    residuals = vectors.numpy()
    models: List[KMeans] = []
    for level, n_clusters in enumerate(levels):
        print(f"Training level {level + 1}/{len(levels)} with {n_clusters} clusters...")
        model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
        assignments = model.fit_predict(residuals)
        models.append(model)
        residuals = residuals - model.cluster_centers_[assignments]
    return models


def semantic_id_path(models: Sequence[KMeans], vector: torch.Tensor) -> Tuple[int, ...]:
    """Infer the hierarchical cluster path for a single embedding."""

    residual = vector.numpy().copy()
    path: List[int] = []
    for model in models:
        cluster = int(model.predict(residual.reshape(1, -1))[0])
        path.append(cluster)
        residual = residual - model.cluster_centers_[cluster]
    return tuple(path)


# Keep the hierarchy shallow enough for quick Colab runs. 3x32 clusters give plenty of
# expressivity (32^3 combinations) without blowing up training time.
hierarchy = [32, 32, 32]
vq_models = train_hierarchical_kmeans(movie_embeddings, hierarchy)

movie_semantic_ids: Dict[int, Tuple[int, ...]] = {}
for movie_id, vector in tqdm(
    zip(catalog_df["movieId"], movie_embeddings),
    total=len(catalog_df),
    desc="Assigning semantic IDs",
):
    movie_semantic_ids[int(movie_id)] = semantic_id_path(vq_models, vector)


# ---------------------------------------------------------------------------
# 5. Convert user histories into semantic ID sequences
# ---------------------------------------------------------------------------

def build_user_sequences(
    ratings: pd.DataFrame, max_sequence_len: int = 20, min_items: int = 5
) -> List[List[int]]:
    """Return truncated sequences of movieIds for users with at least `min_items`."""

    sequences: List[List[int]] = []
    for _, group in ratings.groupby("userId"):
        movie_ids = group["movieId"].tolist()
        # Remove accidental duplicates while keeping chronological order.
        deduped: List[int] = []
        seen: set[int] = set()
        for mid in movie_ids:
            if mid not in seen:
                deduped.append(int(mid))
                seen.add(int(mid))
        if len(deduped) < min_items:
            continue
        sequences.append(deduped[-max_sequence_len:])
    return sequences


raw_sequences = build_user_sequences(ratings_df)
print(f"Training sequences: {len(raw_sequences):,}")


def build_token_vocab(paths: Iterable[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    """Map each unique semantic path to a stable token id (padding=0)."""

    vocab: Dict[Tuple[int, ...], int] = {}
    for path in paths:
        if path not in vocab:
            vocab[path] = len(vocab) + 1
    return vocab


token_vocab = build_token_vocab(movie_semantic_ids.values())


def encode_sequence(sequence: Sequence[int], vocab: Dict[Tuple[int, ...], int]) -> List[int]:
    """Translate movieIds into semantic token ids using the shared vocabulary."""

    return [vocab[movie_semantic_ids[mid]] for mid in sequence]


encoded_sequences = [encode_sequence(seq, token_vocab) for seq in raw_sequences]
max_len = max(len(seq) for seq in encoded_sequences)

# Build autoregressive training pairs (X -> next semantic token).
inputs, targets = [], []
for seq in encoded_sequences:
    input_seq = seq[:-1]
    target_seq = seq[1:]
    pad_len = max_len - 1 - len(input_seq)
    inputs.append(input_seq + [0] * pad_len)
    targets.append(target_seq + [0] * pad_len)

inputs_tensor = torch.tensor(inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)

train_dataset = Dataset.from_dict({"input_ids": inputs_tensor, "labels": targets_tensor})


# ---------------------------------------------------------------------------
# 6. Define a compact semantic decoder (Transformer)
# ---------------------------------------------------------------------------


@dataclass
class SemanticConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class SemanticDecoder(nn.Module):
    """Minimal Transformer decoder for next-token prediction over semantic IDs."""

    def __init__(self, config: SemanticConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None):
        embeddings = self.token_emb(input_ids)
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        decoded = self.transformer(embeddings, embeddings, tgt_mask=mask)
        logits = self.output_proj(decoded)
        loss = None
        if target_ids is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )
        return logits, loss


config = SemanticConfig(vocab_size=len(token_vocab) + 1)
model = SemanticDecoder(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# 7. Lightweight training loop
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 30,
) -> None:
    """Run a simple full-batch training routine suitable for small datasets."""

    model.train()
    batch_inputs = inputs.to(device)
    batch_targets = targets.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, loss = model(batch_inputs, batch_targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f}")


train_model(model, optimizer, inputs_tensor, targets_tensor)


# ---------------------------------------------------------------------------
# 8. Helper to generate recommendations using semantic IDs
# ---------------------------------------------------------------------------


def recommend_next(
    seed_movies: Sequence[int],
    top_k: int = 10,
) -> List[Tuple[List[int], float]]:
    """Return top-k semantic token predictions mapped back to MovieLens ids."""

    model.eval()
    with torch.no_grad():
        seq_tokens = [token_vocab[movie_semantic_ids[mid]] for mid in seed_movies]
        if len(seq_tokens) < 1:
            raise ValueError("Seed sequence must contain at least one movie.")
        padded = seq_tokens + [0] * (max_len - 1 - len(seq_tokens))
        tensor_input = torch.tensor([padded], device=device)
        logits, _ = model(tensor_input)
        next_token_logits = logits[0, len(seq_tokens) - 1]
        probabilities = next_token_logits.softmax(-1)
        top_scores, top_indices = probabilities.topk(top_k + 1)
        results: List[Tuple[List[int], float]] = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
            if idx == 0:
                continue
            matching = [mid for mid, path in movie_semantic_ids.items() if token_vocab[path] == idx]
            results.append((matching, score))
            if len(results) == top_k:
                break
        return results


# Choose a seed user and reveal all but the final interaction to the model.
example_sequence = raw_sequences[0]
seed = example_sequence[:-1]
ground_truth = example_sequence[-1]

print("\nSeed movies:")
for mid in seed:
    row = catalog_df.loc[catalog_df["movieId"] == mid].iloc[0]
    print(f"- {row['title']} ({row['genres']})")

recommendations = recommend_next(seed, top_k=5)

print("\nModel suggestions (semantic clusters mapped to movies):")
for rank, (movie_ids, score) in enumerate(recommendations, start=1):
    readable = ", ".join(
        catalog_df.loc[catalog_df["movieId"].isin(movie_ids)]["title"].head(3).tolist()
    )
    print(f"Top {rank}: {readable} (p={score:.3f})")

gt_title = catalog_df.loc[catalog_df["movieId"] == ground_truth, "title"].iloc[0]
print(f"\nGround-truth next movie in this user's history: {gt_title}")


print("\nDone! You now have semantic IDs trained on MovieLens latest-small.")

