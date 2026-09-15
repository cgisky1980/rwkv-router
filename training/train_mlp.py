"""智能路由 MLP 分类头训练脚本（v4：hidden + prev_tier one-hot 输入）。

复刻 rwkv-router 论文路径 `paper/scripts/03_classification.py`（test_acc=0.9325）：
mean-pooled hidden -> 标准化 -> Linear -> GELU -> LayerNorm -> Dropout -> Linear。

输入：extract_router_features（rwkv-rsv example）产出的 features.jsonl，
     每行 {"idx", "tier", "prev_tier"?, "hidden": [f32, ...]}（prev_tier 缺省
     = None → one-hot index 4）。加载后统一拼接 5 维 one-hot → input_dim = D+5。
输出：router_head.json（v4 契约：含 base_dim=D；Rust RouterHead.forward 由
     prev_tier 参数拼 one-hot，见 head.rs）。权重布局契约见
     client/src/crates/core/src/agent/routing/head.rs。

用法：
    uv run train_mlp.py [--data data/features.jsonl data/zh_features.jsonl
                         data/slices_train_features.jsonl]
                        [--context-ratio 0.5] [--out router_head_v4.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 42
HIDDEN_DIM = 256
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 30
BATCH_SIZE = 256
CLIP = 1.0


class MlpClassifier(nn.Module):
    """与 Rust RouterHead.forward 逐算子对应的分类头。

    Linear(input_dim, hidden) -> GELU -> LayerNorm -> Dropout -> Linear(hidden, 4)
    """

    def __init__(self, input_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


PREV_TIER_DIM = 5  # one-hot: R0-R3 + None（首轮/未知，与 Rust head.rs 契约一致）
PREV_TIER_NONE_IDX = 4


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """返回 (X_hidden, y, prev_tiers)。prev_tiers[i] = 0-3 或 None。"""
    hiddens: list[np.ndarray] = []
    tiers: list[int] = []
    prevs: list = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hiddens.append(np.asarray(row["hidden"], dtype=np.float32))
            tiers.append(int(row["tier"]))
            pt = row.get("prev_tier")
            prevs.append(int(pt) if pt is not None else None)
    X = np.stack(hiddens)
    y = np.asarray(tiers, dtype=np.int64)
    assert X.ndim == 2 and y.shape[0] == X.shape[0]
    assert set(np.unique(y)).issubset({0, 1, 2, 3}), f"unexpected tiers: {np.unique(y)}"
    return X, y, prevs


def append_prev_onehot(X: np.ndarray, prevs: list) -> np.ndarray:
    """hidden [N, D] ‖ prev_tier one-hot [N, 5] -> [N, D+5]（v4 head 输入）。"""
    n = X.shape[0]
    onehot = np.zeros((n, PREV_TIER_DIM), dtype=np.float32)
    for i, p in enumerate(prevs):
        onehot[i, p if p is not None and p < 4 else PREV_TIER_NONE_IDX] = 1.0
    return np.concatenate([X, onehot], axis=1)


def stratified_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 分层切分（train/dev/test），按类轮流分配，seed 固定。"""
    rng = np.random.default_rng(SEED)
    idx_train, idx_dev, idx_test = [], [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_dev = int(n * 0.15)
        n_test = int(n * 0.15)
        idx_test.extend(cls_idx[:n_test])
        idx_dev.extend(cls_idx[n_test : n_test + n_dev])
        idx_train.extend(cls_idx[n_test + n_dev :])
    return np.asarray(idx_train), np.asarray(idx_dev), np.asarray(idx_test)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i : i + BATCH_SIZE]
        yb = y[i : i + BATCH_SIZE]
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    model.train()
    return correct / len(X)


def export_head(
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    base_dim: int,
    out_path: Path,
) -> None:
    """导出 Rust RouterHead 契约格式的权重 JSON（v4：含 base_dim，输入 =
    base_dim hidden + 5 维 prev_tier one-hot；Rust 端 forward 拼接 one-hot）。

    Linear 权重取 nn.Linear.weight（[out, in] 行主序），Rust 端 y = x @ W^T + b。
    """
    lin1 = model.net[0]
    ln = model.net[2]
    lin2 = model.net[4]
    payload = {
        "version": 1,
        "input_dim": base_dim + PREV_TIER_DIM,
        "base_dim": base_dim,
        "hidden_dim": HIDDEN_DIM,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "w1": lin1.weight.detach().cpu().numpy().astype(float).ravel().tolist(),
        "b1": lin1.bias.detach().cpu().numpy().astype(float).tolist(),
        "ln_g": ln.weight.detach().cpu().numpy().astype(float).tolist(),
        "ln_b": ln.bias.detach().cpu().numpy().astype(float).tolist(),
        "w2": lin2.weight.detach().cpu().numpy().astype(float).ravel().tolist(),
        "b2": lin2.bias.detach().cpu().numpy().astype(float).tolist(),
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"exported head -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data/features.jsonl"],
        help="首个文件做 70/15/15 切分；其余文件（数据增强）仅并入 train。",
    )
    parser.add_argument("--out", default="router_head.json")
    parser.add_argument(
        "--context-ratio",
        type=float,
        default=None,
        help="train 中带上下文样本（增强文件里 prev_tier 非 None）的目标占比（0-1）。"
        "镜像线上分布：会话第二轮起必有上下文。None = 不过采样。",
    )
    parser.add_argument(
        "--class-weights",
        default=None,
        help="逐类损失权重，如 1,1,1,3（R3 提权）；None = 均匀。",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="CrossEntropy label smoothing 系数（边界过自信缓解）。",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="MLP 隐层宽度（head.rs 从 JSON 读 hidden_dim，契约安全）；缺省 256。",
    )
    args = parser.parse_args()
    global HIDDEN_DIM
    if args.hidden_dim:
        HIDDEN_DIM = args.hidden_dim

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_path = Path(args.data[0])
    out_path = Path(args.out)

    print(f"loading features: {data_path}")
    X, y, prevs = load_features(data_path)
    base_dim = X.shape[1]
    X = append_prev_onehot(X, prevs)
    input_dim = X.shape[1]
    print(f"samples={len(X)} hidden_dim={base_dim} input_dim={input_dim} "
          f"class_counts={np.bincount(y, minlength=4).tolist()}")

    idx_train, idx_dev, idx_test = stratified_split(y)
    print(f"split: train={len(idx_train)} dev={len(idx_dev)} test={len(idx_test)}")

    # 导出切分索引：Rust 端到端评测（classify_accuracy）用同一 test 集
    # （索引相对于首个特征文件，即 golden 原始行号）。
    split_path = data_path.parent / "split_idx.json"
    split_path.write_text(
        json.dumps(
            {"train": idx_train.tolist(), "dev": idx_dev.tolist(), "test": idx_test.tolist()}
        ),
        encoding="utf-8",
    )
    print(f"split indices -> {split_path}")

    # 增强特征文件：全部并入 train（不参与 dev/test，保证 held-out 纯净）。
    aug_X, aug_y, aug_prevs = [], [], []
    for extra in args.data[1:]:
        p = Path(extra)
        print(f"loading augmentation: {p}")
        x2, y2, p2 = load_features(p)
        assert x2.shape[1] == base_dim, f"{p}: dim {x2.shape[1]} != {base_dim}"
        aug_X.append(append_prev_onehot(x2, p2))
        aug_y.append(y2)
        aug_prevs.extend(p2)
        n_ctx = sum(1 for v in p2 if v is not None)
        print(f"  +{len(x2)} samples (context={n_ctx}), "
              f"class_counts={np.bincount(y2, minlength=4).tolist()}")
    if aug_X:
        X_full = np.concatenate([X] + aug_X)
        y_full = np.concatenate([y] + aug_y)
        aug_idx = np.arange(len(X), len(X_full))
        idx_train = np.concatenate([idx_train, aug_idx])
        print(f"merged train set: {len(idx_train)}")

        # 上下文样本过采样：镜像线上（第二轮起 100% 带上下文）。
        # 重复因子把 train 中 prev_tier 非 None 的占比推到 --context-ratio。
        if args.context_ratio is not None:
            rng = np.random.default_rng(SEED)
            is_ctx = np.array([p is not None for p in prevs + aug_prevs])
            n_train = len(idx_train)
            train_ctx = int(is_ctx[idx_train].sum())
            train_bare = n_train - train_ctx
            # target: train_ctx * k / (train_ctx * k + train_bare) = ratio
            denom = max(1.0 - args.context_ratio, 1e-6)
            k = (args.context_ratio * train_bare) / (denom * max(train_ctx, 1))
            k = max(1.0, min(k, 3.0))
            if k > 1.05:
                ctx_train_idx = idx_train[is_ctx[idx_train]]
                reps = int((k - 1.0) * len(ctx_train_idx))
                if reps > 0:
                    dup = rng.choice(ctx_train_idx, size=reps, replace=True)
                    idx_train = np.concatenate([idx_train, dup])
                    print(f"context oversample: x{k:.2f} (+{reps}) -> "
                          f"context ratio {is_ctx[idx_train].mean():.2f}")
            else:
                print(f"context ratio already ~{train_ctx / max(n_train, 1):.2f}; no oversample")

    # 合并数据集（无增强时 X_full == X）。
    if not aug_X:
        X_full, y_full = X, y

    # train 集标准化统计量（与 Rust 端导出的 mean/std 一致；含增强/过采样样本）。
    mean = X_full[idx_train].mean(axis=0)
    std = X_full[idx_train].std(axis=0)
    std = np.maximum(std, 1e-6)
    Xn = (X_full - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(Xn.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_full).to(device)
    print(f"device={device}")

    model = MlpClassifier(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.class_weights:
        cw = torch.tensor(
            [float(x) for x in args.class_weights.split(",")], dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
        print(f"class weights: {cw.tolist()}")

    Xtr, ytr = Xt[idx_train], yt[idx_train]
    Xdv, ydv = Xt[idx_dev], yt[idx_dev]
    Xte, yte = Xt[idx_test], yt[idx_test]

    best_dev = 0.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        total_loss = 0.0
        for i in range(0, len(Xtr), BATCH_SIZE):
            sel = perm[i : i + BATCH_SIZE]
            xb, yb = Xtr[sel], ytr[sel]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item() * len(sel)
        scheduler.step()
        dev_acc = evaluate(model, Xdv, ydv)
        print(f"epoch {epoch + 1:2d}/{EPOCHS} loss={total_loss / len(Xtr):.4f} dev_acc={dev_acc:.4f}")
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc = evaluate(model, Xte, yte)
    print(f"\nbest dev_acc={best_dev:.4f}  test_acc={test_acc:.4f}")

    export_head(model, mean, std, base_dim, out_path)
    print("done")


if __name__ == "__main__":
    main()
