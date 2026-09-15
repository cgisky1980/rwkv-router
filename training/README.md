# rwkv-router 训练工作台（分类头训练 / 评估 / 闸门包构建）

自进化路由头的离线训练脚本与**训练文本**。本目录只提供脚本和文本源；所有中间产物
（特征 jsonl、eval_pack、replay_pool、head）由你自己跑脚本生成。

## 目录内容

```
training/
├── build_dataset_v5.py       # 文本 → 多轮切片 + train/eval 切分（v5/v53 流水线）
├── sample_scenarios_v5.py    # 场景抽样生成
├── gen_zh_augment.py         # 中文数据增强（生成）
├── build_zh_aug.py           # 中文增强（构建）
├── convert_capture.py        # ★ 网关捕获库(samples.jsonl) → 训练特征 jsonl
├── train_mlp.py              # 训练 MLP 分类头（torch）
├── build_eval_pack.py        # 特征集 → eval_pack.json（闸门）+ replay_pool.json（回放池）
├── eval_head.py              # 评估头的分层准确率
├── relabel_manual_0b.py      # 人工重标注
├── pyproject.toml / uv.lock  # uv 环境（torch + numpy）
└── data/                     # 训练文本源（~60MB，无任何特征/中间产物）
    ├── scenarios_all_v5.jsonl       # 合并场景文本（36.9MB，多轮对话场景）
    ├── summaries_v5_all.jsonl       # 场景摘要
    ├── split_v5.json                # 切分 manifest（复现 build_dataset_v5）
    ├── slices_v53_train/eval/eval_new.jsonl   # v53 切片（build_dataset_v5 的产物快照）
    ├── all_users_v5.jsonl           # 真实用户查询文本
    ├── numina_aug.jsonl / zh_aug.jsonl / rp_aug.jsonl   # 数据增强文本
    └── mt_bench_questions.jsonl     # MT-Bench 问题
```

## 完整流水线

```
文本 data/scenarios_all_v5.jsonl + summaries_v5_all.jsonl
  │ build_dataset_v5.py                       （切片 + 切分）
  ▼
slices_*_train/eval/eval_new.jsonl
  │ 0.1B 骨干提取 hidden —— 两种来源：
  │   a) 网关采集：serve 起来跑真实流量，capture API 自动存
  │      samples.jsonl（含 hidden_hex）→ convert_capture.py 转特征行
  │   b) 离线批量：宿主内嵌 ClassifyEngine 遍历切片文本提取
  ▼
features jsonl（{hidden: [768 f32], tier: 0-3, prev_tier?}）
  │ train_mlp.py                              （AdamW + 标准化冻结）
  ▼
router_head.json
  │ build_eval_pack.py（特征按类对半切 gate/replay）
  ▼
eval_pack.json + replay_pool.json  →  packs_dir 交给网关（进化闸门）
  │ eval_head.py                              （任一时点评估）
  ▼
rwkv-router evolve（微调 → 闸门 → 备份旧头 → 热部署）
```

## 快速开始

```bash
cd training
uv sync          # torch + numpy

# 1) 网关采集来的样本 → 特征行（serve 跑一段时间真实流量后）
uv run python convert_capture.py --capture ../evolution-data/samples.jsonl \
    --out data/capture_features_0b.jsonl

# 2) 训练（多文件输入：首个 70/15/15 切分，其余并入 train）
uv run python train_mlp.py --data data/capture_features_0b.jsonl --out router_head.json

# 3) 闸门包 + 回放池
uv run python build_eval_pack.py --help   # 按提示传特征文件

# 4) 评估
uv run python eval_head.py --head router_head.json --data data/capture_features_0b.jsonl
```

## 特征行契约（features jsonl）

```json
{"hidden": [0.0123, ...], "tier": 2, "prev_tier": 1}
```

- `hidden`：0.1B 骨干 mean-pooled 状态嵌入，768 维 f32
- `tier`：正确层级 0–3（R0–R3）
- `prev_tier`：上一轮层级（0–3）或省略；会话第二轮起建议携带

## head JSON 契约（Rust 侧 `rwkv-router/src/head.rs` 消费）

| 字段 | 说明 |
|---|---|
| `base_dim` | 骨干嵌入维度（768） |
| `mean` / `std` | 标准化参数（训练后冻结——防遗忘的锚） |
| `w1` (hidden_dim×input_dim) / `b1` | MLP 第一层（GELU） |
| `ln_g` / `ln_b` | LayerNorm 仿射 |
| `w2` (4×hidden_dim) / `b2` | 输出 logits（R0–R3） |

v4 头：输入 = `[hidden, onehot(prev_tier | 4)]`（拼接 5 维 one-hot），即 `input_dim = base_dim + 5`。

## eval_pack.json（闸门包）契约

```json
{
  "version": 1,
  "base_dim": 768,
  "samples": [{ "h": "<base64(f16 LE hidden)>", "t": 2, "p": 1 }]
}
```

`build_eval_pack.py` 从特征集按「每源每类对半」切出 gate / replay 两份互斥样本。进化时新头先过闸门：准确率不低于旧头才部署（备份 + 热重载由 Rust 侧 `evolution` 模块完成）。
