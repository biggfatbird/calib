"""Every claim id cited in the manuscripts exists in the generated matrix."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("paper_v2.md", "MANUSCRIPT_DRAFT.md")


def _known_ids() -> set[str]:
    matrix = (ROOT / "CLAIM_EVIDENCE_MATRIX.md").read_text()
    return set(re.findall(r"^\| ([A-Za-z0-9][A-Za-z0-9_\-]*) \|", matrix, re.M))


def _expand(token: str) -> list[str]:
    if "–" not in token:
        return [token]
    left, right = token.split("–", 1)
    left_match = re.match(r"^(.*?)(\d+)$", left)
    right_match = re.match(r"^(.*?)(\d+)$", right)
    if not left_match or not right_match or left_match.group(1) != right_match.group(1):
        return [token]
    prefix = left_match.group(1)
    start = int(left_match.group(2))
    stop = int(right_match.group(2))
    width = len(left_match.group(2))
    pad = left_match.group(2).startswith("0")
    out = []
    for i in range(start, stop + 1):
        number = f"{i:0{width}d}" if pad else str(i)
        out.append(f"{prefix}{number}")
    return out


def cited_ids(path: Path) -> list[str]:
    found = []
    for group in re.findall(r"\[([^\]]+)\]", path.read_text()):
        if "://" in group or group.startswith("http"):
            continue
        for part in group.split(","):
            token = part.strip()
            if not token or not re.match(r"^[A-Z]", token):
                continue
            found.extend(_expand(token))
    return found


def test_cited_claim_ids_resolve_to_the_matrix():
    known = _known_ids()
    missing = {}
    for name in DOCS:
        ids = cited_ids(ROOT / name)
        assert ids, name
        missing[name] = sorted({cid for cid in ids if cid not in known})
    assert missing == {name: [] for name in DOCS}
