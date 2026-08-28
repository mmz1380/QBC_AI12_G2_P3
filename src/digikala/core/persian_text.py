"""Persian text utilities shared across every phase.

`normalize` uses hazm when it's installed (proper Persian normalizer) and falls
back to a pure-unicode path otherwise, so the notebook still runs on a machine
without hazm. `tokenize` and `extract_price_constraint` back the retriever and the
intent router.
"""
from __future__ import annotations

import re
import unicodedata

# hazm gives a better normalizer, but it's a heavy optional dep; degrade gracefully.
try:
    from hazm import Normalizer as _HazmNormalizer
    _hazm = _HazmNormalizer()
except Exception:                                   # pragma: no cover
    _hazm = None

# Persian/Arabic digits -> ASCII, and unify the Arabic ی/ک variants.
_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]", re.UNICODE)
_INVISIBLE_RE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: object, *, fold_digits: bool = True, drop_emoji: bool = False) -> str:
    """Normalize a Persian string. Returns '' for NaN/None."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).translate(_CHAR_MAP)
    s = _URL_RE.sub(" ", s)
    if drop_emoji:
        s = _EMOJI_RE.sub(" ", s)
    s = _hazm.normalize(s) if _hazm is not None else s
    if fold_digits:
        s = s.translate(_DIGIT_MAP)
    s = _INVISIBLE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokenize(text: object) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def tokenize_norm(text: object) -> list[str]:
    """Fast tokenizer for text that is ALREADY normalized (skips the hazm pass).
    Use for corpus building where inputs are the stored *_norm columns."""
    if text is None or (isinstance(text, float) and text != text):
        return []
    return _TOKEN_RE.findall(str(text))


def is_meaningful(text: object, min_chars: int = 2) -> bool:
    return len(normalize(text).replace("‌", "").strip()) >= min_chars


# ---- price constraint parser (used by the router) -----------------------
_NUM = r"(\d+(?:[./]\d+)?)"
_UNIT = r"(هزار|میلیون|میلیارد|k|K)?"
_CUR = r"(تومان|تومن|تومنی|تومانی|ریال)?"
_UPPER_WORDS = ("زیر", "کمتر", "حداکثر", "سقف", "تا", "ارزان‌تر", "پایین‌تر")
_LOWER_WORDS = ("بالای", "بیشتر", "بالاتر", "حداقل", "کف", "گران‌تر")
_DIRECTIONS = _UPPER_WORDS + _LOWER_WORDS
_PRICE_RE = re.compile(
    rf"(?:(?P<kw>{'|'.join(map(re.escape, _DIRECTIONS))})\s*)?"
    rf"(?P<num>{_NUM})\s*(?P<unit>{_UNIT})\s*{_CUR}")


def _unit_mult(u: str | None) -> float:
    return {"هزار": 1e3, "k": 1e3, "K": 1e3, "میلیون": 1e6, "میلیارد": 1e9}.get(u, 1.0)


def extract_price_constraint(text: object, toman_to_rial: int = 10) -> dict:
    """Pull {price_min, price_max} (in Rials) from a Persian query.

    A bare number with no direction word ("زیر"/"بالای"/...) is ignored, and the
    amount is treated as Tomans unless the user wrote "ریال" explicitly.
    """
    t = normalize(text)
    out: dict = {}
    for m in _PRICE_RE.finditer(t):
        kw = (m.group("kw") or "").strip()
        if not kw:
            continue
        val = float(m.group("num").replace("/", ".")) * _unit_mult(m.group("unit"))
        if (m.group(4) or "") != "ریال":            # group 4 is the currency
            val *= toman_to_rial
        if kw in _UPPER_WORDS:
            out["price_max"] = min(out.get("price_max", val), val)
        else:
            out["price_min"] = max(out.get("price_min", val), val)
    return out


def format_toman(value: object) -> str:
    """Rials -> a human 'Toman' string, or 'نامشخص' when missing."""
    if value is None or (isinstance(value, float) and value != value):
        return "نامشخص"
    try:
        return f"{int(float(value)) // 10:,}"
    except (TypeError, ValueError):
        return "نامشخص"
