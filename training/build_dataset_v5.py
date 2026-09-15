"""Build the v5 context-aware training dataset: REAL per-turn conversation slices.

v5 redo (see .trae/documents/router-head-v5-data-plan.md):
  - Single merged batch: scenarios_all_v5.jsonl (old 4 batches + v5 resample,
    gid = global line index) + summaries_v5_all.jsonl (ALL summaries regenerated
    with the FIXED sample kernels — the old batches were generated under the
    buggy rwkv_sample kernel: non-power-of-2 tree reduction orphans + wrong
    top-K threshold, which polluted summaries with echo/repetition garbage).
  - New split seed ("v5"): scenario-level 85/15 by first-user-text hash;
    split manifest written to split_v5.json for reproducibility.
  - Strengthened summary_ok: additionally reject prompt-echo patterns
    (leading "(none)"/"Response to"/"User:", embedded "\\nUser:"/"\\nRequest:"
    /"\\nQuestion:") — degenerate generations from the small summarizer model.
  - eval_new subset: eval slices from v5-new scenarios ONLY (gid >= OLD_COUNT,
    never seen by the v4 head) — clean v4-vs-v5 comparison benchmark.

Per-turn tier labeling / rolling summary chain / prev_tier semantics are
IDENTICAL to v4 (comparability).
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data" / "multiturn"
OUT_TRAIN = DATA / "slices_v5_train.jsonl"
OUT_EVAL = DATA / "slices_v5_eval.jsonl"
OUT_EVAL_NEW = DATA / "slices_v5_eval_new.jsonl"
OUT_EVAL_POOLS = DATA / "eval_summary_pools_v5.json"
OUT_SPLIT = DATA / "split_v5.json"

# 单一合并批：gid = scenarios_all_v5.jsonl 行号（旧 4 批 0..2122 + v5 新采样 2123..）。
SCEN = DATA / "scenarios_all_v5.jsonl"
SUMM = DATA / "summaries_v5_all.jsonl"
# 旧批次场景数（v5 新采样起始 gid）——eval_new 子集边界。
OLD_SCENARIO_COUNT = 2123

TRAIN_RATIO = 85  # hash % 100 < TRAIN_RATIO -> train
SPLIT_SEED = "v5"

CONFIRM_RE = re.compile(
    r"^(好的?|好呀|嗯+|哦+|明白|了解|知道了?|继续|接着说?|请继续|继续吧|再来|说下去|"
    r"ok|okay|o\.?k\.?|got ?it|great|perfect|nice|good|thanks|thank you|cool|sure|yes|"
    r"go ?on|continue|next|keep going|please continue)[.!。!？?\s]*$",
    re.I,
)
FAILURE_CONT_RE = re.compile(
    r"还是报错|还是不行|仍然|依然|还是失败|再次失败|换了?个错|不行了|"
    r"still (crash|fail|error|not work|broken)|fails? again|crash(es|ed)? again|"
    r"error changed|different error|stack ?trace changed|still not work",
    re.I,
)
DEBUG_RE = re.compile(
    r"error|exception|traceback|stack ?trace|segfault|crash|\bbug\b|\bdebug\b|"
    r"not working|fails?|报错|错误|异常|崩溃|调试|失败|修复|不行",
    re.I,
)
CODE_RE = re.compile(r"```|\bdef \b|\bfunction\b|SELECT .* FROM|</?\w+>|=>")
CHITCHAT_RE = re.compile(
    r"哈哈|嘻嘻|无聊|心情|开心|难过|讲个|笑话|晚安|早安|吃饭|好玩|游戏|电影|音乐|"
    r"\b(fun|joke|bored|weekend|hobby|movie|music)\b",
    re.I,
)

MAX_USER_CHARS = 2000


def summary_ok(s: str) -> bool:
    """v4 质量过滤 + v5 回声拒绝（prompt-echo 退化生成）。"""
    if len(s) < 30 or len(s.split()) < 6:
        return False
    weird = sum(1 for c in s if 0x2500 <= ord(c) <= 0x27BF or 0x1F000 <= ord(c) <= 0x1FAFF)
    if not (weird < 2 and weird / max(len(s), 1) <= 0.02):
        return False
    # 回声拒绝：摘要复述了 prompt 模板结构（小摘要模型的退化模式）。
    head = s.lstrip()[:24]
    if head.startswith("(none)") or head.startswith("Response to") or head.startswith("User:"):
        return False
    if "\nUser:" in s or "\nRequest:" in s or "\nQuestion:" in s:
        return False
    return True


def label_turn(user: str, prev_label: int | None, scenario_tier: int, is_computer: bool) -> int:
    u = user.strip()
    # 1. 纯确认/短回应 → 继承上一轮（turn 0 继承对话背景）。上下文决定含义。
    if len(u) <= 24 and CONFIRM_RE.match(u):
        return prev_label if prev_label is not None else scenario_tier
    # 2. 故障延续 → R3（无论背景）。
    if FAILURE_CONT_RE.search(u):
        return 3
    # 3. computer 对话中本轮带 debug 关键词 → R3。
    if is_computer and DEBUG_RE.search(u):
        return 3
    # 4. 明确代码/结构化产出 → R2。
    if CODE_RE.search(u):
        return 2
    # 5. 短消息 + 闲聊词 → R0。
    if len(u) <= 60 and CHITCHAT_RE.search(u):
        return 0
    # 6. 其余 → 对话背景 tier（与 v4 一致）。
    return scenario_tier


def load_batch(scen_path: Path, summ_path: Path):
    """返回 (scenarios, sums)：gid = 场景文件行号（与 batch_summarize 对齐）。"""
    scenarios = []
    for line in scen_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # 与 batch_summarize 的跳过语义对齐（都跳过解析失败行，gid 索引一致）。
            continue
        scenarios.append((row["turns"], int(row["tier"])))
    sums: dict[tuple[int, int], str] = {}
    if summ_path.exists():
        for line in summ_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # 生成进行中的文件尾部可能是半行（BufWriter flush 边界），跳过。
                continue
            sums[(int(r["scenario"]), int(r["turn"]))] = r["summary"]
    return scenarios, sums


def split_of(first_user: str) -> str:
    h = int(hashlib.md5((SPLIT_SEED + first_user).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if h < TRAIN_RATIO else "eval"


def main() -> None:
    train_rows, eval_rows, eval_new_rows = [], [], []
    eval_pools: dict[int, list[str]] = defaultdict(list)
    stats = Counter()
    split_manifest: dict[str, list[int]] = {"train": [], "eval": []}

    scenarios, sums = load_batch(SCEN, SUMM)
    print(f"[v5] {SCEN.name}: {len(scenarios)} scenarios, {len(sums)} summaries")

    for gid, (turns, scenario_tier) in enumerate(scenarios):
        first_user = turns[0]["user"]
        is_computer = scenario_tier >= 2
        side = split_of(first_user)
        split_manifest[side].append(gid)
        is_new = gid >= OLD_SCENARIO_COUNT
        prev_label: int | None = None
        prev_summary: str | None = None
        for ti, turn in enumerate(turns):
            user = turn["user"][:MAX_USER_CHARS]
            label = label_turn(user, prev_label, scenario_tier, is_computer)
            summary = sums.get((gid, ti))
            sum_ok = summary is not None and summary_ok(summary)
            if ti == 0:
                text = user
                row = {"text": text, "tier": label, "prev_tier": None}
            else:
                if sum_ok and prev_summary is not None:
                    text = f"Summary: {prev_summary}\nRequest: {user}"
                else:
                    # 摘要缺失/被过滤 → 裸请求 + prev_tier（线上失败分布）。
                    text = user
                row = {"text": text, "tier": label, "prev_tier": prev_label}
            (train_rows if side == "train" else eval_rows).append(row)
            if side == "eval" and is_new:
                eval_new_rows.append(row)
            stats[side] += 1
            if row["prev_tier"] is not None:
                stats[f"{side}_ctx"] += 1
            else:
                stats[f"{side}_bare"] += 1
            # 滚动链：摘要被过滤时沿用更早的成功摘要（batch_summarize 同语义）。
            if sum_ok:
                prev_summary = summary
            prev_label = label
            # eval 侧摘要池（boundary 压力评估用，与训练零重叠）。
            if side == "eval" and sum_ok:
                eval_pools[scenario_tier].append(summary)

    # 切分清单（可复现）+ 泄漏断言。
    assert not (set(split_manifest["train"]) & set(split_manifest["eval"])), "split leakage!"
    OUT_SPLIT.write_text(
        json.dumps(
            {
                "seed": SPLIT_SEED,
                "train_ratio": TRAIN_RATIO,
                "old_scenario_count": OLD_SCENARIO_COUNT,
                "train": split_manifest["train"],
                "eval": split_manifest["eval"],
            }
        ),
        encoding="utf-8",
    )

    OUT_TRAIN.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    OUT_EVAL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8",
    )
    OUT_EVAL_NEW.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_new_rows) + "\n",
        encoding="utf-8",
    )
    OUT_EVAL_POOLS.write_text(
        json.dumps({str(k): v for k, v in sorted(eval_pools.items())}, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[v5] train: {len(train_rows)} slices -> {OUT_TRAIN.name}")
    print(f"[v5] eval:  {len(eval_rows)} slices -> {OUT_EVAL.name}")
    print(f"[v5] eval_new (v5 scenarios only, clean vs v4): {len(eval_new_rows)} -> {OUT_EVAL_NEW.name}")
    print(f"[v5] stats: {dict(sorted(stats.items()))}")
    print(f"[v5] tier dist train: {dict(sorted(Counter(r['tier'] for r in train_rows).items()))}")
    print(f"[v5] tier dist eval:  {dict(sorted(Counter(r['tier'] for r in eval_rows).items()))}")
    print(f"[v5] eval pools: { {k: len(v) for k, v in sorted(eval_pools.items())} }")


if __name__ == "__main__":
    main()
