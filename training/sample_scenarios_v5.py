"""Sample REAL multi-turn conversations into router scenarios (v5).

Large-scale resample for v5 data redo (see .trae/documents/router-head-v5-data-plan.md):
  - ShareGPT only (MT-Bench exhausted by earlier batches; dedup via --exclude).
  - Target ~8000 new scenarios, ~2000 per tier, --max-turns 6 (deep rolling, as v2c).
  - R0 fallback: strict chitchat heuristic first; if a (kind,lang) bucket is short,
    relax to short-no-code conversations (avg_user < 40) to fill the quota.

Usage:
  uv run --with pyarrow python sample_scenarios_v5.py
"""

import argparse
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).parent / "data" / "multiturn"
RAW = DATA / "raw"
OUT = DATA / "scenarios_v5.jsonl"

MAX_TURNS = 6          # same as v2c: deeper rolling summaries
MAX_MSG_CHARS = 2000   # cap per message (prefill cost control)
MIN_FIRST_USER = 8     # skip empty/greeting-only conversations

# ---- quotas: (source, lang, tier) -> n scenarios (target ~2000/tier overall) ----
# R0: common zh/en ~1000 each (chitchat; fallback fills shortfall)
# R1: common zh/en ~1000 each (default bucket, plenty of candidates)
# R2: computer zh/en ~700 each + common-with-code ~600 total
# R3: computer zh/en ~1000 each (debug)
SHAREGPT_QUOTA = {
    ("common", "zh", 0): 1000, ("common", "en", 0): 1000,
    ("common", "zh", 1): 1000, ("common", "en", 1): 1000,
    ("computer", "zh", 2): 700, ("computer", "en", 2): 700,
    ("common", "zh", 2): 300, ("common", "en", 2): 300,
    ("computer", "zh", 3): 1000, ("computer", "en", 3): 1000,
}

CHITCHAT_ZH = re.compile(r"聊天|无聊|心情|开心|难过|陪我|讲个|笑话|晚安|早安|吃饭|玩|游戏|电影|音乐|喜欢|爱|朋友|放假|周末")
CHITCHAT_EN = re.compile(r"\b(chat|bored|feeling|joke|funny|weekend|hobby|movie|music|game|play|friend|love|miss)\b", re.I)
DEBUG_RE = re.compile(
    r"error|exception|traceback|stack ?trace|segmentation|segfault|core ?dump|"
    r"memory leak|deadlock|crash|\bbug\b|\bdebug\b|not working|doesn'?t work|fails?|"
    r"报错|错误|异常|崩溃|调试|失败|不工作|没用|无效|段错误|内存泄漏|死锁|修复",
    re.I,
)
CODE_RE = re.compile(r"```|\bdef \b|\bfunction\b|\bclass \w+|SELECT .* FROM|</?\w+>|=>|\{\s*\}")


def clean_msg(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()[:MAX_MSG_CHARS]


def valid_turns(turns: list[dict], max_turns: int) -> bool:
    if not (2 <= len(turns) <= max_turns):
        return False
    if len(turns[0]["user"]) < MIN_FIRST_USER:
        return False
    return all(t["user"] and t["assistant"] for t in turns)


def sharegpt_tier(turns: list[dict], kind: str) -> int:
    full = "\n".join(t["user"] + "\n" + t["assistant"] for t in turns)
    avg_user = sum(len(t["user"]) for t in turns) / len(turns)
    has_code = bool(CODE_RE.search(full))
    if kind == "computer":
        n_debug = len(DEBUG_RE.findall(full))
        if n_debug >= 2 or (n_debug >= 1 and len(turns) >= 3):
            return 3
        return 2
    # common: no-code conversations
    if has_code:
        return 2
    if avg_user < 60 and (CHITCHAT_ZH.search(full) or CHITCHAT_EN.search(full)):
        return 0
    return 1


def is_relaxed_r0(turns: list[dict]) -> bool:
    """Fallback R0: short no-code chats (fills quota when strict chitchat runs short)."""
    full = "\n".join(t["user"] for t in turns)
    if CODE_RE.search(full):
        return False
    avg_user = sum(len(t["user"]) for t in turns) / len(turns)
    return avg_user < 40


def load_exclude(paths: list[Path]) -> set[str]:
    """First-user texts of already-sampled scenarios (cross-batch dedup)."""
    excl: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = row.get("turns") or []
                if turns:
                    excl.add(turns[0]["user"])
    return excl


def load_sharegpt(path: Path, kind: str, lang: str, quota: dict[tuple, int],
                  rng: random.Random, excl: set[str], max_turns: int):
    """Collect strict-tier candidates, plus a relaxed-R0 bucket for shortfall filling."""
    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    relaxed_r0: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = row.get("conversation") or []
            turns = []
            for pair in conv:
                u = clean_msg(str(pair.get("human") or ""))
                a = clean_msg(str(pair.get("assistant") or ""))
                if u or a:
                    turns.append({"user": u, "assistant": a})
            turns = turns[:max_turns]
            if not valid_turns(turns, max_turns):
                continue
            if turns[0]["user"] in excl:
                continue
            tier = sharegpt_tier(turns, kind)
            item = {"turns": turns, "tier": tier, "source": f"sharegpt/{kind}_{lang}"}
            buckets[tier].append(item)
            if tier == 1 and kind == "common" and is_relaxed_r0(turns):
                relaxed_r0.append(item)
    out = []
    for t in (0, 1, 2, 3):
        need = quota.get((kind, lang, t), 0)
        pool = buckets[t]
        rng.shuffle(pool)
        take = pool[:need]
        out.extend(take)
        note = ""
        if t == 0 and len(take) < need:
            # R0 shortfall: fill from relaxed bucket (short no-code chats).
            rng.shuffle(relaxed_r0)
            fill = [x for x in relaxed_r0 if x not in take][: need - len(take)]
            for x in fill:
                x["tier"] = 0
            out.extend(fill)
            note = f" (+{len(fill)} relaxed)"
        print(f"[sharegpt {kind}_{lang}] tier{t}: {len(pool)} candidates -> "
              f"{min(need, len(pool))} strict{note}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument(
        "--exclude", nargs="+",
        default=[str(DATA / n) for n in
                 ("scenarios.jsonl", "scenarios_v2.jsonl", "scenarios_v2b.jsonl", "scenarios_v2c.jsonl")],
        help="previous scenarios jsonl(s); skip conversations with the same first user msg",
    )
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = ap.parse_args()

    out_path = Path(args.out)
    excl = load_exclude([Path(p) for p in args.exclude])
    print(f"[sample] seed={args.seed} max_turns={args.max_turns} "
          f"exclude={len(excl)} first-user keys")

    rng = random.Random(args.seed)
    scenarios = []
    for (kind, lang), _paths in {
        ("common", "zh"): [RAW / "common_zh_70k.jsonl"],
        ("common", "en"): [RAW / "common_en_70k.jsonl"],
        ("computer", "zh"): [RAW / "computer_zh_26k.jsonl"],
        ("computer", "en"): [RAW / "computer_en_26k.jsonl"],
    }.items():
        for p in _paths:
            if not p.exists():
                print(f"[warn] missing {p.name}, skipped")
                continue
            scenarios.extend(load_sharegpt(p, kind, lang, SHAREGPT_QUOTA, rng, excl, args.max_turns))

    rng.shuffle(scenarios)
    with out_path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps({"turns": s["turns"], "tier": s["tier"]}, ensure_ascii=False) + "\n")

    dist = {}
    turns_dist = {}
    for s in scenarios:
        dist[s["tier"]] = dist.get(s["tier"], 0) + 1
        turns_dist[len(s["turns"])] = turns_dist.get(len(s["turns"]), 0) + 1
    est = sum(len(s["turns"]) for s in scenarios)
    print(f"\nwrote {len(scenarios)} scenarios -> {out_path}")
    print(f"tier dist: {dict(sorted(dist.items()))}")
    print(f"turns dist: {dict(sorted(turns_dist.items()))}")
    print(f"estimated summaries (one per turn): {est}")

    # 断言：与旧批次零重叠、四 tier ≥1500（配额不足时及早暴露）。
    written = load_exclude([out_path])
    overlap = len(written & excl)
    assert overlap == 0, f"dedup failed: {overlap} first-user keys overlap with old batches"
    for t in range(4):
        assert dist.get(t, 0) >= 1500, f"tier{t} only {dist.get(t, 0)} < 1500"
    print("checks passed: zero overlap with old batches, all tiers >= 1500")


if __name__ == "__main__":
    main()
