"""EuroCrops HCAT class harmonisation for transfer training and interoperable class definitions.

The Hierarchical Crop and Agriculture Taxonomy (HCAT; Schneider et al., 2023, EuroCrops) gives every crop a
ten-digit code whose two-digit pairs encode the hierarchy, e.g. ``33 01 01 01 01`` = crop type / arable /
cereal / common soft wheat / winter. A *harmonisation table* maps HCAT codes to the engine's own class
names, so that parcels from any EuroCrops country (Slovenia, Croatia, Austria, ...) can be relabelled into the
classes of a study (and vice versa for export).

Table format (CSV, one rule per row; see ``configs/hcat/``)::

    class_name,hcat_code,match,hcat_name,note
    maize,3301010600,exact,grain_maize_corn_popcorn,
    grassland,3302000000,subtree,pasture_meadow_grassland_grass,all permanent grassland codes

``match`` is ``exact`` (only this code) or ``subtree`` (this code and every descendant, i.e. all codes that
start with its significant prefix after removing trailing ``00`` pairs). Exact rules win over subtree rules;
among subtree rules the most specific (longest prefix) wins. A row with an empty ``hcat_code`` documents
an engine class without an HCAT equivalent (e.g. water, built-up). Codes not matched by any rule map to
``None`` and are reported, never silently guessed.

The HCAT taxonomy is distributed by EuroCrops under CC-BY-SA-4.0; tables derived from it carry that
licence (see ``configs/hcat/LICENSE-HCAT.md``).
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MATCH_MODES = ("exact", "subtree")


@dataclass(frozen=True)
class Rule:
    class_name: str
    hcat_code: str | None
    match: str = "exact"
    hcat_name: str | None = None
    note: str | None = None

    @property
    def prefix(self) -> str:
        """Significant prefix of the code: trailing ``00`` pairs removed (``3302000000`` -> ``3302``)."""
        return significant_prefix(self.hcat_code or "")


def significant_prefix(code: str) -> str:
    code = normalise_code(code) or ""
    while len(code) > 2 and code.endswith("00"):
        code = code[:-2]
    return code


def normalise_code(code) -> str | None:
    """Ten-digit HCAT code as a string, or None for empty / missing values (floats such as 3.3e9 accepted)."""
    if code is None:
        return None
    if isinstance(code, float):
        if code != code:   # NaN
            return None
        code = f"{code:.0f}"
    text = str(code).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) != 10:
        raise ValueError(f"Invalid HCAT code {code!r}: expected ten digits")
    return text


def validate_rules(rules: Iterable[Rule]) -> list[Rule]:
    rules = list(rules)
    seen: dict[tuple[str, str], str] = {}
    for r in rules:
        if not r.class_name:
            raise ValueError("Every HCAT rule needs a class_name")
        if r.match not in MATCH_MODES:
            raise ValueError(f"Rule for {r.class_name!r}: match must be one of {MATCH_MODES}")
        if r.hcat_code is None:
            continue
        key = (r.match, r.hcat_code)
        if key in seen and seen[key] != r.class_name:
            raise ValueError(f"HCAT code {r.hcat_code} ({r.match}) maps to both {seen[key]!r} and {r.class_name!r}")
        seen[key] = r.class_name
    return rules


def read_rules(path: str | Path) -> list[Rule]:
    """Read a harmonisation CSV (class_name,hcat_code[,match][,hcat_name][,note]); '#' lines are comments."""
    with Path(path).open(encoding="utf8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.lstrip().startswith("#")))
    if rows and not {"class_name", "hcat_code"} <= set(rows[0]):
        raise ValueError(f"{path}: header must contain class_name and hcat_code")
    return validate_rules(
        Rule(class_name=r["class_name"].strip(), hcat_code=normalise_code(r.get("hcat_code")),
             match=(r.get("match") or "exact").strip() or "exact",
             hcat_name=(r.get("hcat_name") or "").strip() or None, note=(r.get("note") or "").strip() or None)
        for r in rows)


def rules_from_spec(spec: dict) -> list[Rule]:
    """Rules from a mapping ``{class: {"exact": [codes], "subtree": [codes]}}`` (benchmark protocols)."""
    rules = []
    for name, entry in spec.items():
        if isinstance(entry, (list, tuple)):
            entry = {"exact": entry}
        for mode in MATCH_MODES:
            for code in entry.get(mode, ()) or ():
                rules.append(Rule(str(name), normalise_code(code), mode))
        if not any(entry.get(m) for m in MATCH_MODES):
            raise ValueError(f"Class {name!r} needs at least one HCAT code")
    return validate_rules(rules)


def classify(code, rules: list[Rule]) -> str | None:
    """Engine class for one HCAT code (exact rule first, then the most specific subtree rule)."""
    code = normalise_code(code)
    if code is None:
        return None
    for r in rules:
        if r.match == "exact" and r.hcat_code == code:
            return r.class_name
    best, best_len = None, -1
    for r in rules:
        if r.match == "subtree" and r.hcat_code and code.startswith(r.prefix) and len(r.prefix) > best_len:
            best, best_len = r.class_name, len(r.prefix)
    return best


def harmonise(codes: Iterable, rules: list[Rule]) -> pd.Series:
    """Vectorised `classify` (memoised per unique code); unmatched or malformed codes become None."""
    codes = pd.Series(list(codes), dtype=object)

    def safe(c):
        try:
            return classify(c, rules)
        except ValueError:
            return None
    lookup = {c: safe(c) for c in codes.dropna().unique()}
    return codes.map(lambda c: lookup.get(c) if c is not None and c == c else None).astype(object)


def coverage_report(codes: Iterable, rules: list[Rule], names: Iterable | None = None) -> dict:
    """Which source codes were mapped / left unmapped, with counts (for documenting transfer labels)."""
    codes = pd.Series(list(codes), dtype=object).astype(str)
    mapped = harmonise(codes, rules)
    frame = pd.DataFrame({"code": codes, "class": mapped})
    if names is not None:
        frame["name"] = list(names)
    unmatched = frame[frame["class"].isna()]
    top = unmatched.groupby("code", dropna=False).size().sort_values(ascending=False).head(20)
    return {"n": int(len(frame)), "mapped": int(frame["class"].notna().sum()),
            "per_class": {str(k): int(v) for k, v in frame["class"].value_counts().items()},
            "top_unmapped_codes": {str(k): int(v) for k, v in top.items()}}


def export_mapping(rules: list[Rule]) -> dict:
    """``{class: {"code", "name"}}`` for `geocrop.export.stac_item` (first rule per class is representative)."""
    out: dict[str, dict] = {}
    for r in rules:
        out.setdefault(r.class_name, {"code": r.hcat_code, "name": r.hcat_name})
    return out
