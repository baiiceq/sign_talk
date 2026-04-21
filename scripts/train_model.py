#!/usr/bin/env python3
"""
单手手势 BiLSTM 训练脚本（PyTorch）

流程：
1) 从 data/gestures/<label>/*.npy 读取序列
2) 分层切分 train/val/test
3) 训练 BiLSTM
4) 导出 checkpoint(.pt)、TorchScript(.ts)、标签映射与训练指标
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练单手手势 BiLSTM 模型")
    parser.add_argument("--data-dir", type=str, default="data/gestures", help="数据目录（label 子目录）")
    parser.add_argument("--output-dir", type=str, default="models", help="模型输出目录")
    parser.add_argument("--model-name", type=str, default="gesture_model", help="模型名称前缀")
    parser.add_argument("--sequence-length", type=int, default=30, help="序列长度")
    parser.add_argument("--feature-dim", type=int, default=68, help="特征维度")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM 隐层维度")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM 层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集占比")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集占比")
    parser.add_argument("--min-samples-per-label", type=int, default=100, help="每个标签最少样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GestureSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_scale = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * direction_scale, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


@dataclass
class LoadedData:
    x: np.ndarray
    y: np.ndarray
    label_to_idx: dict[str, int]


def load_sequences(data_dir: Path, sequence_length: int, feature_dim: int, min_samples_per_label: int) -> LoadedData:
    label_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        raise FileNotFoundError(f"未找到标签目录: {data_dir}")

    label_to_idx: dict[str, int] = {}
    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for idx, label_dir in enumerate(label_dirs):
        label = label_dir.name
        files = sorted(label_dir.glob("*.npy"))
        if len(files) < min_samples_per_label:
            raise ValueError(
                f"标签 '{label}' 样本不足: {len(files)} < {min_samples_per_label}. "
                "请先补充采集后再训练。"
            )

        label_to_idx[label] = idx
        for f in files:
            seq = np.load(f).astype(np.float32)
            if seq.shape != (sequence_length, feature_dim):
                raise ValueError(
                    f"样本维度不匹配: {f}, got {seq.shape}, "
                    f"expected {(sequence_length, feature_dim)}"
                )
            x_list.append(seq)
            y_list.append(idx)

    x = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return LoadedData(x=x, y=y, label_to_idx=label_to_idx)


def create_splits(x: np.ndarray, y: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio + test_ratio >= 0.5:
        raise ValueError("val_ratio + test_ratio 不应 >= 0.5")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=val_ratio + test_ratio, random_state=seed, stratify=y
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=relative_test, random_state=seed, stratify=y_tmp
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_pred: list[int] = []
    all_true: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * y_batch.size(0)
            total_correct += (preds == y_batch).sum().item()
            total_count += y_batch.size(0)

            all_pred.extend(preds.detach().cpu().numpy().tolist())
            all_true.extend(y_batch.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc, all_true, all_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_sequences(
        data_dir=data_dir,
        sequence_length=args.sequence_length,
        feature_dim=args.feature_dim,
        min_samples_per_label=args.min_samples_per_label,
    )
    idx_to_label = {v: k for k, v in data.label_to_idx.items()}

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_splits(
        data.x, data.y, args.val_ratio, args.test_ratio, args.seed
    )

    train_loader = DataLoader(
        GestureSequenceDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        GestureSequenceDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        GestureSequenceDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(data.label_to_idx),
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_acc = 0.0
    best_checkpoint = output_dir / f"{args.model_name}.pt"

    print("=" * 72)
    print(f"训练设备: {device}")
    print(f"类别数: {len(data.label_to_idx)}")
    print(f"数据规模: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"输出模型: {best_checkpoint}")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * y_batch.size(0)
            running_correct += (preds == y_batch).sum().item()
            running_count += y_batch.size(0)

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": args.feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": len(data.label_to_idx),
                    "dropout": args.dropout,
                    "bidirectional": True,
                    "label_to_idx": data.label_to_idx,
                    "sequence_length": args.sequence_length,
                    "best_val_acc": best_val_acc,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
        digits=4,
        zero_division=0,
    )
    print("\n=== Test Result ===")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(report)

    scripted_path = output_dir / f"{args.model_name}.ts"
    example = torch.randn(1, args.sequence_length, args.feature_dim, device=device)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(scripted_path))

    labels_path = output_dir / f"{args.model_name}_labels.json"
    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    labels_path.write_text(
        json.dumps(
            {
                "label_to_idx": data.label_to_idx,
                "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "train_size": int(len(y_train)),
                "val_size": int(len(y_val)),
                "test_size": int(len(y_test)),
                "sequence_length": args.sequence_length,
                "feature_dim": args.feature_dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"训练完成。checkpoint: {best_checkpoint}")
    print(f"TorchScript: {scripted_path}")
    print(f"标签映射: {labels_path}")
    print(f"指标文件: {metrics_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
