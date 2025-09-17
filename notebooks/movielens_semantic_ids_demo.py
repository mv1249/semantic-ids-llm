"""Fine-tune Qwen on MovieLens semantic IDs (Colab-friendly demo).

This single-file notebook mirrors the training recipe described in the README but keeps the
implementation compact enough to paste into one Google Colab cell.  It:

1. Downloads the MovieLens *latest-small* dataset (~100k ratings).
2. Encodes movie metadata with a lightweight sentence-transformer.
3. Learns hierarchical semantic IDs via residual K-Means.
4. Adds those semantic ID tokens to a small Qwen model.
5. Fine-tunes Qwen with LoRA so it can predict the next semantic ID from a user history.
6. Generates recommendations by sampling the fine-tuned model.

Usage (drop the whole cell into Colab):

```
%pip -q install pandas==2.2.2 sentence-transformers==3.0.1 scikit-learn==1.5.1 \
    torch==2.3.1 transformers==4.41.2 datasets==2.20.0 peft==0.10.0 accelerate==0.31.0

%run movielens_semantic_ids_demo.py
```

The training loop uses a 0.5B parameter Qwen variant plus LoRA, which fits comfortably on a
free Colab T4 GPU when using the default batch size and sequence length.  The code is heavily
commented so you can adapt it or share snippets in a blog / LinkedIn post.
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# 1. Reproducibility helpers
# ---------------------------------------------------------------------------

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def seed_everything(seed: int = SEED) -> None:
    """Ensure deterministic behaviour across `random` and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")


# ---------------------------------------------------------------------------
# 2. Download and load MovieLens metadata
# ---------------------------------------------------------------------------

DATA_ROOT = Path("./data/movielens-latest-small")
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(destination: Path = DATA_ROOT) -> Path:
    """Fetch MovieLens latest-small (cached on repeated runs)."""

    if (destination / "ml-latest-small").exists():
        return destination / "ml-latest-small"

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.with_suffix(".zip")
    print("Downloading MovieLens latest-small (~9MB)...")
    torch.hub.download_url_to_file(DATASET_URL, archive_path)

    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    return destination / "ml-latest-small"


def load_catalog(dataset_dir: Path) -> pd.DataFrame:
    """Read movies and add a short text description used for embeddings."""

    movies = pd.read_csv(dataset_dir / "movies.csv")
    movies["text"] = movies.apply(
        lambda row: f"{row['title']} Genres: {row['genres'].replace('|', ', ')}.", axis=1
    )
    return movies


def load_positive_interactions(dataset_dir: Path, min_rating: float = 3.5) -> pd.DataFrame:
    """Keep only ratings that look like positive user-item interactions."""

    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    positive = ratings[ratings["rating"] >= min_rating].copy()
    positive.sort_values(["userId", "timestamp"], inplace=True)
    return positive


dataset_path = download_movielens()
catalog_df = load_catalog(dataset_path)
ratings_df = load_positive_interactions(dataset_path)

print(f"Movies: {len(catalog_df):,} | Positive interactions: {len(ratings_df):,}")


# ---------------------------------------------------------------------------
# 3. Embed item metadata with a sentence transformer
# ---------------------------------------------------------------------------

encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
with torch.inference_mode():
    movie_embeddings = encoder.encode(
        catalog_df["text"].tolist(), convert_to_tensor=True, device=device
    ).cpu()


# ---------------------------------------------------------------------------
# 4. Learn hierarchical semantic IDs (residual K-Means)
# ---------------------------------------------------------------------------

def train_hierarchical_kmeans(vectors: torch.Tensor, levels: Sequence[int]) -> List[KMeans]:
    """Train a small stack of K-Means quantizers, subtracting residuals at each layer."""

    residuals = vectors.numpy()
    models: List[KMeans] = []
    for level, n_clusters in enumerate(levels):
        print(f"Training quantizer level {level + 1}/{len(levels)} with {n_clusters} clusters...")
        model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
        assignments = model.fit_predict(residuals)
        models.append(model)
        residuals = residuals - model.cluster_centers_[assignments]
    return models


def semantic_id_path(models: Sequence[KMeans], vector: torch.Tensor) -> Tuple[int, ...]:
    """Map an embedding to a tuple of cluster indices (one per level)."""

    residual = vector.numpy().copy()
    path: List[int] = []
    for model in models:
        cluster = int(model.predict(residual.reshape(1, -1))[0])
        path.append(cluster)
        residual = residual - model.cluster_centers_[cluster]
    return tuple(path)


def path_to_token(path: Sequence[int]) -> str:
    """Create a readable token representing a semantic ID path."""

    joined = "-".join(str(level) for level in path)
    return f"<SID:{joined}>"


hierarchy = [16, 16]  # two levels keep the demo fast while creating varied IDs
vq_models = train_hierarchical_kmeans(movie_embeddings, hierarchy)

movie_to_sid: Dict[int, Tuple[int, ...]] = {}
movie_to_token: Dict[int, str] = {}
token_to_titles: Dict[str, List[str]] = {}

for movie_id, embedding, title in zip(
    catalog_df["movieId"], movie_embeddings, catalog_df["title"]
):
    path = semantic_id_path(vq_models, embedding)
    token = path_to_token(path)
    movie_to_sid[int(movie_id)] = path
    movie_to_token[int(movie_id)] = token
    token_to_titles.setdefault(token, []).append(title)

unique_tokens = sorted(set(movie_to_token.values()))
print(f"Unique semantic ID tokens: {len(unique_tokens)}")


# ---------------------------------------------------------------------------
# 5. Prepare sequential training examples
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """Simple container for user history text and the target semantic ID token."""

    history_tokens: str
    target_token: str

    @property
    def prompt(self) -> str:
        return f"History: {self.history_tokens}\nNext: {self.target_token}"


def build_examples(
    interactions: pd.DataFrame,
    mapping: Dict[int, str],
    min_history: int = 2,
    max_history: int = 5,
) -> List[TrainingExample]:
    """Slide over each user timeline and create history → next-token pairs."""

    examples: List[TrainingExample] = []
    for user_id, group in interactions.groupby("userId"):
        item_ids = group["movieId"].tolist()
        if len(item_ids) <= min_history:
            continue
        for index in range(min_history, len(item_ids)):
            history_slice = item_ids[max(0, index - max_history): index]
            history_tokens = " ".join(mapping[mid] for mid in history_slice)
            target_token = mapping[item_ids[index]]
            examples.append(TrainingExample(history_tokens, target_token))
    return examples


examples = build_examples(ratings_df, movie_to_token)
print(f"Training examples: {len(examples):,}")

# Keep a lightweight subset for speedy demos (tweak as needed in Colab).
MAX_EXAMPLES = 4000
if len(examples) > MAX_EXAMPLES:
    random.shuffle(examples)
    examples = examples[:MAX_EXAMPLES]
    print(f"Subsampled to {len(examples)} examples for a quick run.")

text_samples = [ex.prompt for ex in examples]
dataset = Dataset.from_dict({"text": text_samples})
splits = dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = splits["train"]
valid_dataset = splits["test"]

print(
    f"Train: {len(train_dataset):,} | Validation: {len(valid_dataset):,}"
)


# ---------------------------------------------------------------------------
# 6. Load Qwen tokenizer & model, then extend vocabulary with semantic IDs
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

new_tokens = [token for token in unique_tokens if token not in tokenizer.get_vocab()]
if new_tokens:
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {len(new_tokens)} semantic ID tokens to the tokenizer.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Configure a lightweight LoRA adapter on attention + projection layers.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# 7. Tokenization + Trainer setup
# ---------------------------------------------------------------------------

def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
    return tokenizer(batch["text"], truncation=True, max_length=160)


train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
valid_dataset = valid_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qwen_movielens_semantic_ids",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

print("Starting LoRA fine-tuning...")
trainer.train()


# ---------------------------------------------------------------------------
# 8. Generate a recommendation for a random validation example
# ---------------------------------------------------------------------------

semantic_token_set = set(unique_tokens)

sample_example = random.choice(examples)
prompt = f"History: {sample_example.history_tokens}\nNext:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
model.eval()
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
decoded = tokenizer.decode(generated_tokens, skip_special_tokens=False)
print("\nPrompt:\n", prompt)
print("Generated completion:\n", decoded)

predicted_token = None
for candidate in semantic_token_set:
    if candidate in decoded:
        predicted_token = candidate
        break

if predicted_token:
    titles = token_to_titles[predicted_token]
    print("\nTop semantic token:", predicted_token)
    print("Candidate movies sharing this token:")
    for title in titles[:5]:
        print(" -", title)
else:
    print("\nNo semantic token found in the generated output. Try sampling again or fine-tuning longer.")

print("\nGround-truth next token:", sample_example.target_token)
print("Ground-truth movie choices:")
for title in token_to_titles[sample_example.target_token][:5]:
    print(" -", title)

print("\nDone! You can now experiment with different prompts or export the LoRA adapters.")
