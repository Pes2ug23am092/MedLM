from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

DATASET_ID = "jhlee0619/mpib"
REQUIRED_COLUMNS = ["vector", "user_query", "rule", "parent_sample_id", "metadata"]
OPTIONAL_COLUMNS = ["scenario", "labels"]
HARD_NEG_RE = re.compile(r"overdose|lethal dose|toxic|opioid|suicid", re.IGNORECASE)
DEFAULT_FILES = ["data/train.jsonl", "data/validation.jsonl", "data/test.jsonl"]

LOGIN_HELP = """
Access to jhlee0619/mpib was denied. It is a GATED dataset. Do this once:
  1. Create a free account at huggingface.co and a READ token at
     huggingface.co/settings/tokens
  2. Open huggingface.co/datasets/jhlee0619/mpib and click "Agree / Request access"
  3. In Colab run:   from huggingface_hub import notebook_login; notebook_login()
     (or pass --hf-token hf_xxx, or set the HF_TOKEN environment variable)
"""

_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "validation", "valid": "validation", "validation": "validation", "dev": "validation",
    "test": "test", "testing": "test",
}


@dataclass
class MPIBRow:
    vector: str
    query: str
    rule: str
    parent: str
    source: str | None
    split: str
    hard: bool


def _norm_split(x) -> str | None:
    if x is None:
        return None
    return _SPLIT_ALIASES.get(str(x).strip().lower())


def _norm_source(x) -> str | None:
    if x is None:
        return None
    s = re.sub(r"[\s_\-]", "", str(x).lower())
    return {"medqa": "MedQA", "pubmedqa": "PubMedQA"}.get(s, str(x))


def _norm_vector(x) -> str:
    s = str(x).strip().lower().replace("\u2019", "'").replace("_", "").replace("-", "")
    table = {
        "v0": "V0", "v0'": "V0'", "v0prime": "V0'", "v0p": "V0'",
        "v1": "V1", "v2": "V2",
    }
    return table.get(s, str(x).strip())


def _norm_rule(x) -> str:
    m = re.search(r"R\s*([1-6])", str(x or ""), re.IGNORECASE)
    return f"R{m.group(1)}" if m else ""


def _as_dict(m) -> dict:
    if isinstance(m, dict):
        return m
    if isinstance(m, str):
        try:
            d = json.loads(m)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _list_data_files(repo: str, token: str | None) -> list[str]:
    """Find the .jsonl data files in the repo; fall back to the usual names."""
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset", token=token)
    except Exception:  # noqa: BLE001
        return DEFAULT_FILES
    found = sorted(f for f in files if f.endswith(".jsonl"))
    return found or DEFAULT_FILES


def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: bad JSON on line {line_no}: {e}") from e
    return out


def load_mpib(
    repo: str = DATASET_ID, config: str | None = None, token: str | None = None
) -> list[MPIBRow]:
    """Load MPIB by reading the raw JSONL files directly.

    We avoid datasets.load_dataset because MPIB's nested metadata has varying
    fields across rows, which makes Arrow schema casting fail.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if config:
        print(f"[mpib] note: --config {config!r} is ignored (raw JSONL loader).")

    files = _list_data_files(repo, token)
    print(f"[mpib] data files: {files}")

    rows: list[MPIBRow] = []
    for fname in files:
        try:
            local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", token=token)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(LOGIN_HELP + f"\nOriginal error: {type(e).__name__}: {e}") from e
        records = _read_jsonl(local)
        file_split = _norm_split(re.sub(r"\.jsonl?$", "", fname.rsplit("/", 1)[-1]))

        all_cols: set[str] = set()
        for rec in records:
            all_cols.update(rec.keys())
        print(f"[mpib] '{fname}': {len(records)} rows; columns = {sorted(all_cols)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in all_cols]
        if missing:
            raise ValueError(
                f"'{fname}' is missing required columns {missing}. Found: {sorted(all_cols)}"
            )
        absent_opt = [c for c in OPTIONAL_COLUMNS if c not in all_cols]
        if absent_opt:
            print(f"[mpib] note: optional columns not present: {absent_opt}")

        for ex in records:
            meta = _as_dict(ex.get("metadata"))
            split = _norm_split(meta.get("split")) or file_split or "unknown"
            query = ex.get("user_query")
            parent = ex.get("parent_sample_id")
            rows.append(
                MPIBRow(
                    vector=_norm_vector(ex.get("vector")),
                    query=str(query) if query is not None else "",
                    rule=_norm_rule(ex.get("rule")),
                    parent=str(parent) if parent not in (None, "") else "",
                    source=_norm_source(meta.get("source")),
                    split=split,
                    hard=bool(HARD_NEG_RE.search(str(query or ""))),
                )
            )

    print(f"[mpib] loaded {len(rows)} rows. Counts by (vector, source, split):")
    counts = Counter((r.vector, r.source, r.split) for r in rows)
    for key in sorted(counts, key=lambda k: tuple(str(x) for x in k)):
        print(f"    {key}: {counts[key]}")
    return rows


def partition(rows: list[MPIBRow]) -> dict[str, list[MPIBRow]]:
    """pos = V1 (MedQA); neg = V0 MedQA; neg_pubmed = V0 PubMedQA; v0p = V0'. V2 dropped."""
    pos = [r for r in rows if r.vector == "V1" and r.source in ("MedQA", None)]
    dropped = sum(1 for r in rows if r.vector == "V1") - len(pos)
    neg = [r for r in rows if r.vector == "V0" and r.source in ("MedQA", None)]
    neg_pubmed = [r for r in rows if r.vector == "V0" and r.source == "PubMedQA"]
    v0p = [r for r in rows if r.vector == "V0'"]
    n_v2 = sum(1 for r in rows if r.vector == "V2")
    unknown = Counter(r.vector for r in rows if r.vector not in ("V0", "V0'", "V1", "V2"))
    print(
        f"[mpib] positives V1: {len(pos)} (dropped non-MedQA V1: {dropped}); "
        f"negatives V0-MedQA: {len(neg)} (hard: {sum(r.hard for r in neg)}); "
        f"V0-PubMedQA: {len(neg_pubmed)}; V0': {len(v0p)} (excluded, reported separately); "
        f"V2: {n_v2} (excluded)"
    )
    if unknown:
        print(f"[mpib] WARNING: unrecognised vector values ignored: {dict(unknown)}")
    return {"pos": pos, "neg": neg, "neg_pubmed": neg_pubmed, "v0p": v0p}
