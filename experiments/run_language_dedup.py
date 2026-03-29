#!/usr/bin/env python3
"""Entry point for per-language dedup rate analysis."""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import get_hf_token, preflight_check, cache_language
from src.language_dedup import run_language_dedup, aggregate_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/experiment_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def action_preflight(config: dict) -> None:
    """Phase 1: Pre-flight check for all languages."""
    token = get_hf_token()
    hf_dataset = config["data"]["hf_dataset"]
    languages = config["data"]["languages"]

    results = {}
    for lang in tqdm(list(languages.keys()), desc="Preflight", unit="lang"):
        data_dir = languages[lang]["data_dir"]
        passed = preflight_check(hf_dataset, data_dir, lang, token)
        results[lang] = "PASS" if passed else "FAIL"

    print("\n=== Preflight Results ===")
    for lang, status in results.items():
        print(f"  {lang:15s} {status}")

    n_pass = sum(1 for s in results.values() if s == "PASS")
    print(f"\n{n_pass}/{len(results)} languages passed preflight.")


def action_cache(config: dict) -> None:
    """Phase 2: Cache all language data to Parquet."""
    token = get_hf_token()
    hf_dataset = config["data"]["hf_dataset"]
    cache_dir = config["data"]["cache_dir"]
    n_files = config["data"]["n_files_per_language"]
    min_len = config["data"]["min_file_length"]
    languages = config["data"]["languages"]

    for lang in tqdm(list(languages.keys()), desc="Languages (cache)", unit="lang"):
        data_dir = languages[lang]["data_dir"]
        cache_language(
            hf_dataset=hf_dataset,
            data_dir=data_dir,
            language=lang,
            cache_dir=cache_dir,
            n_files=n_files,
            min_file_length=min_len,
            token=token,
        )


def action_run(config: dict, language: str | None = None) -> None:
    """Phase 3: Run dedup for all (or one) language(s)."""
    cache_dir = config["data"]["cache_dir"]
    results_dir = config["output"]["results_dir"]
    minhash_config = config["minhash"]
    languages = config["data"]["languages"]

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if language:
        if language not in languages:
            logger.error("Unknown language: %s", language)
            sys.exit(1)
        run_language_dedup(language, cache_dir, results_dir, minhash_config)
    else:
        lang_list = list(languages.keys())
        for lang in tqdm(lang_list, desc="Languages (dedup)", unit="lang"):
            try:
                run_language_dedup(lang, cache_dir, results_dir, minhash_config)
            except Exception as e:
                logger.error("Failed to process %s: %s", lang, e)
                continue

    # Aggregate results
    aggregate_results(results_dir, list(languages.keys()), minhash_config)


def main():
    parser = argparse.ArgumentParser(
        description="MinHash LSH deduplication analysis on The Stack v1"
    )
    parser.add_argument(
        "--action",
        choices=["preflight", "cache", "run"],
        required=True,
        help="Phase to execute: preflight, cache, or run",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Run dedup for a single language (only with --action run)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to experiment config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.action == "preflight":
        action_preflight(config)
    elif args.action == "cache":
        action_cache(config)
    elif args.action == "run":
        action_run(config, args.language)

    # Force exit: datasets streaming leaves background threads alive
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
