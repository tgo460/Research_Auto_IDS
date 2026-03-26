from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


SplitEntry = Union[str, Dict[str, Any], "SplitArtifactRef"]


@dataclass(frozen=True)
class SplitArtifactRef:
    path: str
    row_start: Optional[int] = None
    row_stop: Optional[int] = None

    def display_name(self) -> str:
        if self.row_start is None and self.row_stop is None:
            return self.path
        start = 0 if self.row_start is None else int(self.row_start)
        stop = "end" if self.row_stop is None else int(self.row_stop)
        return f"{self.path}[{start}:{stop}]"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path}
        if self.row_start is not None:
            out["row_start"] = int(self.row_start)
        if self.row_stop is not None:
            out["row_stop"] = int(self.row_stop)
        return out


def parse_split_entry(entry: SplitEntry) -> SplitArtifactRef:
    if isinstance(entry, SplitArtifactRef):
        return entry
    if isinstance(entry, str):
        return SplitArtifactRef(path=entry)
    if not isinstance(entry, dict):
        raise TypeError(f"split entry must be a string or object, got {type(entry)!r}")

    path = entry.get("path") or entry.get("file")
    if not path or not isinstance(path, str):
        raise ValueError("split entry object must include a string 'path'")

    row_start = entry.get("row_start", entry.get("start"))
    row_stop = entry.get("row_stop", entry.get("stop"))
    if row_start is not None:
        row_start = int(row_start)
    if row_stop is not None:
        row_stop = int(row_stop)
    if row_start is not None and row_start < 0:
        raise ValueError(f"row_start must be >= 0 for {path}")
    if row_stop is not None and row_stop < 0:
        raise ValueError(f"row_stop must be >= 0 for {path}")
    if row_start is not None and row_stop is not None and row_start >= row_stop:
        raise ValueError(f"row_start must be < row_stop for {path}")
    return SplitArtifactRef(path=path, row_start=row_start, row_stop=row_stop)


def is_attack_split_entry(entry: SplitEntry) -> bool:
    ref = parse_split_entry(entry)
    lowered = ref.path.lower()
    return any(token in lowered for token in ("injected", "attack", "dos", "fuzzy", "gear", "rpm"))


def split_entries_overlap(left: SplitEntry, right: SplitEntry) -> bool:
    a = parse_split_entry(left)
    b = parse_split_entry(right)
    if a.path != b.path:
        return False

    a_start = 0 if a.row_start is None else int(a.row_start)
    b_start = 0 if b.row_start is None else int(b.row_start)
    a_stop = float("inf") if a.row_stop is None else int(a.row_stop)
    b_stop = float("inf") if b.row_stop is None else int(b.row_stop)
    return max(a_start, b_start) < min(a_stop, b_stop)
