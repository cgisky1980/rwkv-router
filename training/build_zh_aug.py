"""中文分档增强集（train-only）：补 zh R2/R3 缺口（en 79% vs zh 69% R2 召回）。

来源（全部真实中文数据）：
  R3 = RUC-AIBOX/OlymMATH ZH-EASY/HARD（奥赛级中文题，~350）
  R2 = jean1/45k_python_code_chinese_instruction（中文编程，2500）
       + hails/agieval-gaokao-mathcloze（高考填空，1500）
  R1 = weitianwen/cmath cmath_test（小学数学中文，2500）

输出：data/multiturn/zh_aug.jsonl（{text, tier}）。去重同 numina 管线。
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

BASE = Path(__file__).parent
MT = BASE / "data" / "multiturn"
OUT = MT / "zh_aug.jsonl"

QUOTA_R2_PY = 2500
QUOTA_R2_GK = 1500
QUOTA_R1 = 2500
MAX_CHARS = 1500
MIN_CHARS = 20
SEED = 20260825


def seen_hashes() -> set[str]:
    seen = set()
    for name in ["slices_v53_train", "slices_v53_eval", "slices_v53_eval_new", "numina_aug"]:
        p = MT / f"{name}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)["text"]
            seen.add(hashlib.md5(t.strip()[:120].encode()).hexdigest())
    return seen


def ok(u: str, zh_ratio: float = 0.25) -> bool:
    if not MIN_CHARS <= len(u) <= MAX_CHARS:
        return False
    n_han = sum(1 for c in u if "\u4e00" <= c <= "\u9fff")
    return n_han / max(len(u), 1) >= zh_ratio


def main() -> None:
    rng = random.Random(SEED)
    seen = seen_hashes()
    internal: set[str] = set()
    # R3 奥赛题 LaTeX 密集，汉字阈值放宽
    ZH_MIN = {1: 0.15, 2: 0.25, 3: 0.10}

    def add(pools: dict[int, list[str]], tier: int, u: str) -> None:
        u = u.strip()
        if not ok(u, ZH_MIN[tier]):
            return
        h = hashlib.md5(u[:120].encode()).hexdigest()
        if h in internal or h in seen:
            return
        internal.add(h)
        pools.setdefault(tier, []).append(u)

    pools: dict[int, list[str]] = {}

    # ---- R3: OlymMATH ZH ----
    for part in ["OlymMATH-ZH-EASY.jsonl", "OlymMATH-ZH-HARD.jsonl"]:
        p = hf_hub_download("RUC-AIBOX/OlymMATH", f"data/{part}", repo_type="dataset")
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("problem") or r.get("question") or ""
            add(pools, 3, u)

    # ---- R2: 45k python chinese ----
    p = hf_hub_download("jean1/45k_python_code_chinese_instruction", "45k_chinese.csv", repo_type="dataset")
    import csv

    with open(p, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            u = row.get("instruction") or row.get("input") or next(iter(row.values()), "")
            add(pools, 2, u)

    # ---- R2: agieval gaokao mathcloze ----
    import pyarrow.parquet as pq

    p = hf_hub_download("hails/agieval-gaokao-mathcloze", "data/test-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p)
    cols = t.column_names
    key = next((c for c in ["query", "question", "problem", "input"] if c in cols), cols[0])
    for u in t.column(key).to_pylist():
        add(pools, 2, u or "")

    # ---- R1: cmath ----
    p = hf_hub_download("weitianwen/cmath", "cmath_test.jsonl", repo_type="dataset")
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        u = r.get("problem") or r.get("question") or ""
        add(pools, 1, u)

    print("pools:", {k: len(v) for k, v in sorted(pools.items())})

    rows = []
    pools[2] = rng.sample(pools[2], min(QUOTA_R2_PY + QUOTA_R2_GK, len(pools[2])))
    pools[1] = rng.sample(pools[1], min(QUOTA_R1, len(pools[1])))
    for tier in (1, 2, 3):
        rows.extend({"text": u, "tier": tier} for u in pools[tier])
    rng.shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} -> {OUT}")
    print(f"tier dist: {dict(sorted(Counter(r['tier'] for r in rows).items()))}")
    for tier in (3, 2, 1):
        sub = [r["text"] for r in rows if r["tier"] == tier]
        print(f"--- R{tier} sample:")
        for u in sub[:3]:
            print("   ", u[:70].replace("\n", " "))


if __name__ == "__main__":
    main()
