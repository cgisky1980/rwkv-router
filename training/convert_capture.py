#!/usr/bin/env python3
"""把 rwkv-router 网关采集的捕获库（samples.jsonl）转成训练特征 jsonl。

流水线位置：文本 →(build_dataset_v5)→ 切片 →(0.1B 提取 hidden，见 README)→
features jsonl →(train_mlp)→ head；本脚本覆盖「网关真实采集」来源：
capture API / `rwkv-router stats` 背后的 samples.jsonl → 特征行。

用法：
    uv run python convert_capture.py \
        --capture ../evolution-data/samples.jsonl \
        --out data/capture_features_0b.jsonl

只保留已标注样本（label 非空）；hidden_hex 为 f16 little-endian hex，
每 4 个 hex 字符（2 字节）解码为一个 f32。
"""

import argparse
import json
import struct
from pathlib import Path


def hex_to_f32(s: str) -> list[float]:
    raw = bytes.fromhex(s)
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True, type=Path, help="网关 samples.jsonl 路径")
    ap.add_argument("--out", required=True, type=Path, help="输出特征 jsonl 路径")
    ap.add_argument("--num-embd", type=int, default=768, help="骨干嵌入维度校验（不符的行跳过）")
    args = ap.parse_args()

    total = kept = skipped_label = skipped_dim = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for line in args.capture.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = row.get("label")
            if label is None:
                skipped_label += 1
                continue
            if row.get("num_embd") != args.num_embd:
                skipped_dim += 1
                continue
            rec = {
                "hidden": hex_to_f32(row["hidden_hex"]),
                "tier": int(label),
            }
            if row.get("prev_tier") is not None:
                rec["prev_tier"] = int(row["prev_tier"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"{args.capture.name}: {total} rows -> {kept} features "
        f"(skipped: {skipped_label} unlabeled, {skipped_dim} dim-mismatch) -> {args.out}"
    )


if __name__ == "__main__":
    main()
