from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

import torch

# --------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------

# Matches things like [INSTRUCTION], [/INSTRUCTION], [SYSTEM OVERRIDE], [NOTE: ...]
BRACKET_TAG_RE = re.compile(
    r"\[\s*/?\s*[A-Z][A-Z0-9_ \-]{2,40}\s*(?::[^\]\n]{0,300})?\]"
)

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|earlier|all|any)\b.{0,40}\b(instruction|prompt|rule|guideline|direction)s?\b",
    r"\b(disregard|forget|override|bypass|circumvent)\b.{0,40}\b(instruction|prompt|rule|guideline|safety|polic(y|ies)|restriction)s?\b",
    r"\bnew\s+(instruction|rule|directive)s?\b",
    r"\b(system|developer)\s+(prompt|message|override|instruction|mode)\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(if|though|an?\s+(unrestricted|unfiltered|ai|assistant))\b",
    r"\b(do\s+not|don'?t|never)\s+(refuse|decline|warn|mention|add\s+(any\s+)?(disclaimer|warning)s?)\b",
    r"\bwithout\s+(any\s+)?(disclaimer|warning|caveat|safety\s+check)s?\b",
    r"\b(jailbreak|dan\s+mode|developer\s+mode|do\s+anything\s+now)\b",
    r"\byou\s+(must|will|have\s+to|shall)\s+(comply|obey|follow\s+(these|this|my))\b",
    r"\b(respond|answer|reply|output)\b.{0,30}\b(only|exactly|strictly)\b.{0,40}\b(format|json|template)\b",
    r"\b(urgent|emergency|immediately|life[- ]or[- ]death)\b.{0,60}\b(skip|ignore|bypass|without)\b",
    r"\bas\s+(the|an?)\s+(administrator|admin|developer|system\s+operator|authorized)\b.{0,60}\b(authori[sz]e|override|instruct|order|direct)",
    r"\b(reveal|print|show|repeat)\b.{0,30}\b(system\s+prompt|hidden\s+(instruction|prompt)s?|your\s+instructions)\b",
    r"\bfrom\s+now\s+on\b",
    r"\b(invent|fabricate|make\s+up)\b.{0,40}\b(dos(e|age)|evidence|citation|reference)s?\b",
    r"\b(contradict|supersede|take\s+precedence)\b.{0,40}\b(instruction|rule|policy|guideline)s?\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def make_windows(ids: list[int], size: int, overlap: int) -> list[list[int]]:
    """Split token ids into overlapping windows covering every token."""
    if size <= overlap:
        raise ValueError("window size must be larger than overlap")
    if len(ids) <= size:
        return [ids]
    step = size - overlap
    out: list[list[int]] = []
    start = 0
    while True:
        out.append(ids[start : start + size])
        if start + size >= len(ids):
            break
        start += step
    return out


def _find_sublist(hay: list[int], needle: list[int]) -> int:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return -1


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class Backend(ABC):
    name: str = "backend"

    @property
    def default_threshold(self) -> float:
        return 0.5

    @abstractmethod
    def score_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        """Return P(malicious)-like score in [0, 1] per (already normalized) text."""


class _RegexBase(Backend):
    def _score_one(self, text: str) -> float:
        raise NotImplementedError

    def score_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        return [self._score_one(t) for t in texts]


class RegexBackend(_RegexBase):
    """Hand-written heuristic baseline. Score = 1 - 0.5**n_patterns_matched."""

    name = "regex"

    def _score_one(self, text: str) -> float:
        n = sum(1 for p in _COMPILED if p.search(text))
        return 1.0 - 0.5**n


class BracketTagBackend(_RegexBase):
    """Only fires on bracketed tags such as [INSTRUCTION]. Artifact check."""

    name = "bracket_tags"

    def _score_one(self, text: str) -> float:
        return 1.0 if BRACKET_TAG_RE.search(text) else 0.0


class HFBackend(Backend):
    def __init__(
        self,
        name: str,
        model_id: str,
        malicious_label: str,
        device: str | None = None,
        max_length: int = 512,
        overlap: int = 64,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.overlap = overlap

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        matches = [i for i, lab in id2label.items() if lab == malicious_label]
        if len(matches) != 1:
            raise ValueError(
                f"Malicious label {malicious_label!r} not found (exactly once) in "
                f"id2label of {model_id}: {id2label}"
            )
        self.malicious_idx = matches[0]

        # Learn the special-token wrapper (e.g. [CLS] ... [SEP]) from a probe,
        # so we do not depend on tokenizer methods that differ across versions.
        probe = self.tokenizer("hello", add_special_tokens=True)["input_ids"]
        plain = self.tokenizer("hello", add_special_tokens=False)["input_ids"]
        pos = _find_sublist(list(probe), list(plain))
        if pos < 0 or not plain:
            raise RuntimeError("Could not infer special-token layout from tokenizer.")
        self.prefix: list[int] = list(probe[:pos])
        self.suffix: list[int] = list(probe[pos + len(plain):])

        self.window = max_length - len(self.prefix) - len(self.suffix)
        if self.overlap >= self.window:
            raise ValueError("overlap must be smaller than the window size")
        self.pad_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )

    @torch.inference_mode()
    def score_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        seqs: list[list[int]] = []
        owners: list[int] = []
        for i, text in enumerate(texts):
            ids = self.tokenizer(
                text, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"]
            if not ids:
                continue
            for w in make_windows(list(ids), self.window, self.overlap):
                full = self.prefix + w + self.suffix
                if len(full) > self.max_length:  # should never happen
                    raise RuntimeError("window exceeds max_length; refusing to truncate")
                seqs.append(full)
                owners.append(i)

        scores = [0.0] * len(texts)
        order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            longest = max(len(seqs[j]) for j in batch)
            input_ids = torch.full((len(batch), longest), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), longest), dtype=torch.long)
            for r, j in enumerate(batch):
                n = len(seqs[j])
                input_ids[r, :n] = torch.tensor(seqs[j], dtype=torch.long)
                attn[r, :n] = 1
            logits = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attn.to(self.device),
            ).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, self.malicious_idx]
            for j, p in zip(batch, probs.cpu().tolist()):
                o = owners[j]
                scores[o] = max(scores[o], p)  # max over windows
        return scores


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_MODEL_SPECS = {
    "prompt_guard_2": ("meta-llama/Llama-Prompt-Guard-2-86M", "LABEL_1"),
    "deberta_v2": ("protectai/deberta-v3-base-prompt-injection-v2", "INJECTION"),
}
BACKEND_NAMES = ("prompt_guard_2", "deberta_v2", "regex", "bracket_tags")


def create_backend(name: str, device: str | None = None) -> Backend:
    if name == "regex":
        return RegexBackend()
    if name == "bracket_tags":
        return BracketTagBackend()
    if name in _MODEL_SPECS:
        model_id, label = _MODEL_SPECS[name]
        return HFBackend(name, model_id, label, device=device)
    raise ValueError(f"Unknown backend {name!r}. Choose from {BACKEND_NAMES}")
