"""智能路由进化闸门评估包生成脚本（eval_pack.json + replay_pool.json）。

从既有 0.1B 特征评估集抽样固化 hidden+标签，供客户端 Rust 侧进化闭环使用：
每次原位微调时，捕获样本与 replay_pool 混合训练（防灾难性遗忘），新旧头再在
eval_pack 上评估，准确率不回退才允许上线。两个包从每源每类样本**互斥切分**
（shuffle 后前半 gate / 后半 replay），保证闸门集未被训练污染。

数据源（全部 _0b = 0.1B 常驻路由模型特征，与部署头 router_head_0b.json 同骨干）：
    - slices_v53_eval_features_0b.jsonl        （clean eval）
    - slices_v53_eval_new_features_0b.jsonl    （clean eval_new）
    - boundary_eval_combined1040_features_0b.jsonl（边界组合）
    - boundary_eval_v5x_features_0b.jsonl      （边界 v5x）

输出格式（两文件相同）：
    {"version": 1, "base_dim": 768, "samples": [
        {"h": "<f16 LE base64>", "t": <tier 0-3>, "p": <prev_tier|null>, "s": "<source>"},
        ...]}
注意：
    - eval_pack 是「冻结」评估集——不得混入运行时捕获样本，不得参与训练；
    - replay_pool 是「回放」训练集——微调时按类均衡混入，缓解大样本量下的遗忘；
    - 换骨干模型（不同 n_embd）时必须用新特征重新生成两包。

用法：
    uv run build_eval_pack.py [--per-source 300] [--head router_head_0b.json] [--no-replay]
    （实验排除源时用 --exclude eval_new,boundary_1040）
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"

SOURCES = {
    "eval": DATA / "slices_v53_eval_features_0b.jsonl",
    "eval_new": DATA / "slices_v53_eval_new_features_0b.jsonl",
    "boundary_1040": DATA / "boundary_eval_combined1040_features_0b.jsonl",
    "boundary_v5x": DATA / "boundary_eval_v5x_features_0b.jsonl",
}


def hidden_to_f16_b64(hidden: list[float]) -> str:
    arr = np.asarray(hidden, dtype=np.float16)  # little-endian on x86
    return base64.b64encode(arr.tobytes()).decode("ascii")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tier = int(row["tier"])
            if tier < 0 or tier > 3:
                continue
            rows.append(
                {
                    "hidden": row["hidden"],
                    "tier": tier,
                    "prev_tier": row.get("prev_tier"),
                    "src": path.stem.replace("_features_0b", ""),
                }
            )
    return rows


def split_gate_replay(
    rows: list[dict], per_class: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """每源每类 shuffle 后前半给 gate、后半给 replay（互斥）。"""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["tier"]].append(r)
    gate: list[dict] = []
    replay: list[dict] = []
    for tier in sorted(by_class):
        cls = by_class[tier]
        rng.shuffle(cls)
        # 取 2*per_class 上限，对半分；不足时对半（各保底 1）。
        usable = cls[: per_class * 2]
        half = max(len(usable) // 2, 1)
        gate.extend(usable[:half])
        replay.extend(usable[half : half * 2])
    return gate, replay


def head_accuracy(head: dict, samples: list[dict]) -> tuple[float, dict[str, float]]:
    """离线复核：用指定头在 eval_pack 样本上的准确率（信息性，非闸门）。"""
    mean = np.asarray(head["mean"], dtype=np.float32)
    std = np.asarray(head["std"], dtype=np.float32)
    w1 = np.asarray(head["w1"], dtype=np.float32).reshape(head["hidden_dim"], head["input_dim"])
    b1 = np.asarray(head["b1"], dtype=np.float32)
    ln_g = np.asarray(head["ln_g"], dtype=np.float32)
    ln_b = np.asarray(head["ln_b"], dtype=np.float32)
    w2 = np.asarray(head["w2"], dtype=np.float32).reshape(4, head["hidden_dim"])
    b2 = np.asarray(head["b2"], dtype=np.float32)

    is_v4 = "base_dim" in head
    base_dim = head.get("base_dim", head["input_dim"])
    onehot = np.eye(5, dtype=np.float32)

    per_source: dict[str, list[int]] = defaultdict(list)
    hit = 0
    for s in samples:
        h = np.asarray(s["hidden"], dtype=np.float32)
        pt = s["prev_tier"]
        x = np.concatenate([h, onehot[4 if pt is None else int(pt)]]) if is_v4 else h
        x = (x - mean) / np.maximum(std, 1e-6)
        z = w1 @ x + b1
        a = 0.5 * z * (1.0 + np.tanh(0.7978846 * (z + 0.044715 * z**3)))
        mu = a.mean()
        var = ((a - mu) ** 2).mean()
        y = (a - mu) / np.sqrt(var + 1e-5) * ln_g + ln_b
        logits = w2 @ y + b2
        pred = int(np.argmax(logits))
        ok = int(pred == s["tier"])
        hit += ok
        per_source[s["src"]].append(ok)

    acc = hit / max(len(samples), 1)
    detail = {k: sum(v) / len(v) for k, v in per_source.items()}
    return acc, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=300, help="每源每类抽样上限（单侧）")
    ap.add_argument("--head", type=str, default="router_head_0b.json")
    ap.add_argument("--out", type=str, default="eval_pack.json")
    ap.add_argument("--replay-out", type=str, default="replay_pool.json")
    ap.add_argument("--no-replay", action="store_true", help="只生成闸门集")
    ap.add_argument("--exclude", type=str, default="", help="逗号分隔的排除源（实验用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    rng = random.Random(args.seed)
    gate_samples: list[dict] = []
    replay_samples: list[dict] = []
    for name, path in SOURCES.items():
        if name in excluded:
            print(f"[excluded] {name}")
            continue
        if not path.exists():
            print(f"[skip] missing: {path.name}")
            continue
        rows = load_rows(path)
        gate, replay = split_gate_replay(rows, args.per_source, rng)
        print(f"{name}: {len(rows)} rows -> gate {len(gate)} + replay {len(replay)}")
        gate_samples.extend(gate)
        replay_samples.extend(replay)

    if not gate_samples:
        raise SystemExit("no samples collected — check data/ files exist")

    def pack(samples: list[dict]) -> dict:
        return {
            "version": 1,
            "base_dim": len(samples[0]["hidden"]),
            "samples": [
                {
                    "h": hidden_to_f16_b64(s["hidden"]),
                    "t": s["tier"],
                    "p": s["prev_tier"],
                    "s": s["src"],
                }
                for s in samples
            ],
        }

    out_path = HERE / args.out
    out_path.write_text(json.dumps(pack(gate_samples), ensure_ascii=False), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"wrote {out_path} : {len(gate_samples)} samples, {size_mb:.1f} MB")

    replay_path = None
    if not args.no_replay and replay_samples:
        replay_path = HERE / args.replay_out
        replay_path.write_text(
            json.dumps(pack(replay_samples), ensure_ascii=False), encoding="utf-8"
        )
        r_mb = replay_path.stat().st_size / 1024 / 1024
        print(f"wrote {replay_path} : {len(replay_samples)} samples, {r_mb:.1f} MB")

    head_path = HERE / args.head
    if head_path.exists():
        head = json.loads(head_path.read_text(encoding="utf-8"))
        # 信息性复核：f16 量化前后的准确率漂移（应 <0.5%）。
        acc_full, detail_full = head_accuracy(head, gate_samples)
        packed = json.loads(out_path.read_text(encoding="utf-8"))["samples"]
        samples_f16 = [
            {
                "hidden": list(np.frombuffer(base64.b64decode(s["h"]), dtype=np.float16).astype(np.float32)),
                "tier": s["t"],
                "prev_tier": s["p"],
                "src": s["s"],
            }
            for s in packed
        ]
        acc_f16, detail_f16 = head_accuracy(head, samples_f16)
        print(f"baseline acc ({args.head}): full={acc_full:.4f} f16={acc_f16:.4f}")
        for k in sorted(detail_full):
            print(f"  {k}: full={detail_full[k]:.4f} f16={detail_f16.get(k, 0):.4f}")
        if abs(acc_full - acc_f16) > 0.005:
            print("WARNING: f16 quantization drift > 0.5% — reconsider precision")
        if replay_path is not None:
            acc_replay, _ = head_accuracy(head, replay_samples)
            print(f"replay pool acc ({args.head}): {acc_replay:.4f}")


if __name__ == "__main__":
    main()
