"""Core MinHash + LSH logic."""

import re
import logging
from multiprocessing import Pool, cpu_count

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from .union_find import UnionFind

logger = logging.getLogger(__name__)


def preprocess(content: str) -> str:
    """Lowercase and collapse all whitespace runs to a single space."""
    return re.sub(r"\s+", " ", content.lower()).strip()


def shingle(text: str, k: int = 5) -> set[str]:
    """Generate character-level k-grams via sliding window."""
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def compute_minhash(args: tuple) -> tuple[int, MinHash]:
    """Compute MinHash signature for a single file. Used by multiprocessing pool."""
    file_id, content, n_perm, shingle_k = args
    mh = MinHash(num_perm=n_perm)
    text = preprocess(content)
    for s in shingle(text, shingle_k):
        mh.update(s.encode("utf-8"))
    return file_id, mh


def generate_signatures(
    file_ids: list[int],
    contents: list[str],
    n_perm: int,
    shingle_k: int,
    n_workers: int,
    batch_size: int,
    language: str,
) -> list[tuple[int, MinHash]]:
    """Generate MinHash signatures in parallel batches."""
    workers = min(n_workers, cpu_count())
    args = [(fid, c, n_perm, shingle_k) for fid, c in zip(file_ids, contents)]
    signatures = []

    n_batches = (len(args) + batch_size - 1) // batch_size
    with Pool(workers) as pool:
        for i in tqdm(range(n_batches), desc=f"MinHash {language}", unit="batch"):
            batch = args[i * batch_size : (i + 1) * batch_size]
            results = pool.map(compute_minhash, batch)
            signatures.extend(results)

    return signatures


def run_lsh_dedup(
    signatures: list[tuple[int, MinHash]],
    n_perm: int,
    threshold: float,
    repo_names: list[str],
    paths: list[str],
    language: str,
) -> dict:
    """Run LSH indexing + Union-Find deduplication. Returns per-file results."""
    lsh = MinHashLSH(threshold=threshold, num_perm=n_perm)

    # Log LSH band/row configuration
    b = lsh.b
    r = lsh.r
    logger.info("LSH config for %s: b=%d, r=%d (threshold=%.2f, num_perm=%d)",
                language, b, r, threshold, n_perm)

    n_files = len(signatures)

    # Insert all signatures into LSH
    for file_id, sig in tqdm(
        signatures, desc=f"LSH insert {language}", unit="files", total=n_files
    ):
        key = str(file_id)
        try:
            lsh.insert(key, sig)
        except ValueError:
            # Duplicate key — already inserted
            pass

    # Build signature lookup for querying
    sig_dict = {str(fid): sig for fid, sig in signatures}

    # Union-Find: stream pairs directly, never materialize pair list
    uf = UnionFind(n_files)
    for file_id, sig in tqdm(
        signatures, desc=f"Union-Find {language}", unit="files", total=n_files
    ):
        neighbors = lsh.query(sig)
        for neighbor_key in neighbors:
            neighbor_id = int(neighbor_key)
            if neighbor_id != file_id:
                uf.union(file_id, neighbor_id)

    # Extract components
    components = uf.components()

    # For each component, find canonical representative (lowest lexicographic repo/path)
    file_results = {}
    for root, members in components.items():
        component_size = len(members)
        if component_size == 1:
            fid = members[0]
            file_results[fid] = {
                "keep": True,
                "component_id": root,
                "component_size": 1,
            }
        else:
            # Find canonical: lowest lexicographic repository_name/path
            canonical = min(
                members,
                key=lambda fid: f"{repo_names[fid]}/{paths[fid]}",
            )
            for fid in members:
                file_results[fid] = {
                    "keep": fid == canonical,
                    "component_id": root,
                    "component_size": component_size,
                }

    return {
        "file_results": file_results,
        "lsh_bands": b,
        "lsh_rows": r,
        "n_components": len(components),
    }
