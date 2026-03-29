"""Per-language dedup rate analysis."""

import json
import time
import logging
from pathlib import Path
from collections import Counter

from .data_loading import load_cached_data
from .minhash_pipeline import generate_signatures, run_lsh_dedup

logger = logging.getLogger(__name__)


def compute_cluster_distribution(
    component_sizes: list[int], n_files_input: int
) -> dict:
    """Compute cluster size distribution and fraction in large clusters."""
    size_counts = Counter(component_sizes)

    buckets = {
        "size_2": 0,
        "size_3_9": 0,
        "size_10_99": 0,
        "size_100_999": 0,
        "size_1000_plus": 0,
    }

    for size, count in size_counts.items():
        files_in_bucket = size * count
        if size == 1:
            continue  # singletons are not duplicate clusters
        elif size == 2:
            buckets["size_2"] += files_in_bucket
        elif 3 <= size <= 9:
            buckets["size_3_9"] += files_in_bucket
        elif 10 <= size <= 99:
            buckets["size_10_99"] += files_in_bucket
        elif 100 <= size <= 999:
            buckets["size_100_999"] += files_in_bucket
        else:
            buckets["size_1000_plus"] += files_in_bucket

    cluster_size_distribution = {
        k: {"fraction_of_files": v / n_files_input} for k, v in buckets.items()
    }

    # Fraction in large clusters at different thresholds
    fraction_in_large = {}
    for thresh in [10, 100, 1000]:
        files_in_large = sum(
            size * count
            for size, count in size_counts.items()
            if size >= thresh
        )
        fraction_in_large[f"threshold_{thresh}"] = files_in_large / n_files_input

    return {
        "cluster_size_distribution": cluster_size_distribution,
        "fraction_in_large_clusters": fraction_in_large,
    }


def run_language_dedup(
    language: str,
    cache_dir: str,
    results_dir: str,
    minhash_config: dict,
) -> dict | None:
    """Run full dedup pipeline for a single language."""
    result_path = Path(results_dir) / f"language_dedup_{language}.json"
    if result_path.exists():
        logger.info("Result already exists for %s, skipping", language)
        with open(result_path) as f:
            return json.load(f)

    start_time = time.time()

    # Load cached data
    df = load_cached_data(cache_dir, language)
    n_files_input = len(df)
    file_ids = df["file_id"].tolist()
    contents = df["content"].tolist()
    repo_names = df["repository_name"].tolist()
    file_paths = df["path"].tolist()

    # Generate MinHash signatures
    signatures = generate_signatures(
        file_ids=file_ids,
        contents=contents,
        n_perm=minhash_config["n_permutations"],
        shingle_k=minhash_config["shingle_k"],
        n_workers=minhash_config["n_workers"],
        batch_size=minhash_config["batch_size"],
        language=language,
    )

    # Run LSH + Union-Find dedup
    dedup_result = run_lsh_dedup(
        signatures=signatures,
        n_perm=minhash_config["n_permutations"],
        threshold=minhash_config["threshold"],
        repo_names=repo_names,
        paths=file_paths,
        language=language,
    )

    file_results = dedup_result["file_results"]

    # Compute file-level metrics
    n_kept = sum(1 for r in file_results.values() if r["keep"])
    n_removed = n_files_input - n_kept
    dedup_rate = n_removed / n_files_input if n_files_input > 0 else 0.0

    # Compute token-level metrics (whitespace-split)
    total_tokens_input = 0
    total_tokens_kept = 0
    for fid, content in zip(file_ids, contents):
        tokens = len(content.split())
        total_tokens_input += tokens
        if file_results[fid]["keep"]:
            total_tokens_kept += tokens
    total_tokens_removed = total_tokens_input - total_tokens_kept
    token_removal_rate = (
        total_tokens_removed / total_tokens_input if total_tokens_input > 0 else 0.0
    )

    # Compute cluster size distribution
    component_sizes = [r["component_size"] for r in file_results.values()]
    cluster_info = compute_cluster_distribution(component_sizes, n_files_input)

    # Find max component size
    max_component_size = max(component_sizes) if component_sizes else 0
    # Mean component size (over multi-file components only)
    multi_file_sizes = [s for s in component_sizes if s > 1]
    mean_component_size = (
        sum(multi_file_sizes) / len(multi_file_sizes)
        if multi_file_sizes
        else 0.0
    )

    runtime_sec = round(time.time() - start_time, 1)

    result = {
        "language": language,
        "threshold": minhash_config["threshold"],
        "shingle_k": minhash_config["shingle_k"],
        "n_permutations": minhash_config["n_permutations"],
        "lsh_bands": dedup_result["lsh_bands"],
        "lsh_rows": dedup_result["lsh_rows"],
        "n_files_input": n_files_input,
        "n_files_kept": n_kept,
        "n_files_removed": n_removed,
        "dedup_rate": round(dedup_rate, 4),
        "total_tokens_input": total_tokens_input,
        "total_tokens_kept": total_tokens_kept,
        "token_removal_rate": round(token_removal_rate, 4),
        "n_components": dedup_result["n_components"],
        "mean_component_size": round(mean_component_size, 2),
        "max_component_size": max_component_size,
        "cluster_size_distribution": cluster_info["cluster_size_distribution"],
        "fraction_in_large_clusters": cluster_info["fraction_in_large_clusters"],
        "runtime_sec": runtime_sec,
    }

    # Write per-language result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Completed %s: dedup_rate=%.3f, token_removal_rate=%.3f, runtime=%ds",
        language,
        dedup_rate,
        token_removal_rate,
        runtime_sec,
    )
    return result


def aggregate_results(results_dir: str, languages: list[str], minhash_config: dict) -> dict:
    """Aggregate all per-language results into a single JSON."""
    aggregate = {
        "threshold": minhash_config["threshold"],
        "shingle_k": minhash_config["shingle_k"],
        "n_permutations": minhash_config["n_permutations"],
        "languages": {},
    }

    for lang in languages:
        result_path = Path(results_dir) / f"language_dedup_{lang}.json"
        if result_path.exists():
            with open(result_path) as f:
                lang_result = json.load(f)
            # Remove redundant top-level keys for the aggregate
            lang_data = {
                k: v
                for k, v in lang_result.items()
                if k not in ("language", "threshold", "shingle_k", "n_permutations")
            }
            aggregate["languages"][lang] = lang_data
        else:
            logger.warning("No result file for %s", lang)

    output_path = Path(results_dir) / "language_dedup.json"
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    logger.info("Aggregated results written to %s", output_path)
    return aggregate
