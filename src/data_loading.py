"""Stream and cache data from HuggingFace."""

import os
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Set it with: export HF_TOKEN='hf_...'\n"
            "You also need to accept the dataset terms at "
            "https://huggingface.co/datasets/bigcode/the-stack"
        )
    return token


def preflight_check(
    hf_dataset: str, data_dir: str, language: str, token: str
) -> bool:
    """Stream 100 rows to confirm the data_dir key works and content is present."""
    try:
        ds = load_dataset(
            hf_dataset,
            data_dir=data_dir,
            split="train",
            streaming=True,
            token=token,
        )
        valid = 0
        for i, row in enumerate(ds):
            if i >= 100:
                break
            if row.get("content") and len(row["content"]) > 0:
                valid += 1
        if valid == 0:
            logger.error(
                "Preflight FAIL for %s: 0 valid rows out of 100", language
            )
            return False
        logger.info(
            "Preflight PASS for %s: %d/100 rows with non-empty content",
            language,
            valid,
        )
        return True
    except Exception as e:
        logger.error("Preflight FAIL for %s: %s", language, e)
        return False


def cache_language(
    hf_dataset: str,
    data_dir: str,
    language: str,
    cache_dir: str,
    n_files: int,
    min_file_length: int,
    token: str,
) -> Path | None:
    """Stream n_files valid files for a language and write to Parquet."""
    cache_path = Path(cache_dir) / f"stack_{language}_200K.parquet"
    if cache_path.exists():
        logger.info("Cache already exists for %s, skipping: %s", language, cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            hf_dataset,
            data_dir=data_dir,
            split="train",
            streaming=True,
            token=token,
        )
    except Exception as e:
        logger.error("Failed to load dataset for %s: %s", language, e)
        return None

    rows = []
    file_id = 0
    with tqdm(total=n_files, desc=f"Caching {language}", unit="files") as pbar:
        for row in ds:
            if file_id >= n_files:
                break
            content = row.get("content", "")
            if not content or len(content) < min_file_length:
                continue

            # Normalize: strip BOM, normalize line endings
            if content.startswith("\ufeff"):
                content = content[1:]
            content = content.replace("\r\n", "\n").replace("\r", "\n")

            rows.append(
                {
                    "file_id": file_id,
                    "content": content,
                    "repository_name": row.get("repository_name", ""),
                    "path": row.get("path", ""),
                }
            )
            file_id += 1
            pbar.update(1)

    if file_id < n_files:
        logger.warning(
            "Only cached %d/%d files for %s", file_id, n_files, language
        )

    if not rows:
        logger.error("No valid files found for %s", language)
        return None

    table = pa.Table.from_pydict(
        {
            "file_id": [r["file_id"] for r in rows],
            "content": [r["content"] for r in rows],
            "repository_name": [r["repository_name"] for r in rows],
            "path": [r["path"] for r in rows],
        }
    )
    pq.write_table(table, cache_path)
    logger.info("Cached %d files for %s to %s", file_id, language, cache_path)
    return cache_path


def load_cached_data(cache_dir: str, language: str) -> pd.DataFrame:
    """Load cached Parquet file for a language."""
    cache_path = Path(cache_dir) / f"stack_{language}_200K.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    return pd.read_parquet(cache_path)
