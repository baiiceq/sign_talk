#!/usr/bin/env python3
"""
单手手势数据采集脚本

特性：
1. 仅按单手场景采集（默认每条序列 30 帧）
2. 支持标签清单自动生成与自定义
3. 自动估算采集总时长
4. 保存 .npy 与 manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.gesture_detector import MediaPipeDetector, FeatureExtractor


DEFAULT_LABELS = [
    "left_swipe",
    "right_swipe",
    "up_swipe",
    "down_swipe",
    "fist_click",
    "open_palm",
    "thumb_up",
    "thumb_down",
    "ok_sign",
    "peace",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单手手势序列数据采集")
    parser.add_argument("--labels", type=str, default="", help="逗号分隔标签，留空使用默认标签清单")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头设备 ID")
    parser.add_argument("--output-dir", type=str, default="data/gestures", help="输出目录")
    parser.add_argument("--sequences-per-label", type=int, default=400, help="每标签采集序列数")
    parser.add_argument("--sequence-length", type=int, default=30, help="每条序列帧数")
    parser.add_argument("--capture-fps", type=float, default=20.0, help="采集有效帧率（用于时间估算）")
    parser.add_argument("--warmup-seconds", type=float, default=2.0, help="每个标签开始前倒计时")
    parser.add_argument("--rest-seconds", type=float, default=0.3, help="每条序列采集后的短暂停顿")
    parser.add_argument("--strict-single-hand", action="store_true", help="启用后：仅在检测到一只手时才计入帧")
    return parser.parse_args()


def get_labels(raw: str) -> list[str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    if labels:
        return labels
    return DEFAULT_LABELS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def estimate_collection_minutes(
    label_count: int,
    sequences_per_label: int,
    sequence_length: int,
    capture_fps: float,
    warmup_seconds: float,
    rest_seconds: float,
) -> float:
    per_sequence = (sequence_length / max(capture_fps, 1.0)) + max(rest_seconds, 0)
    total_seconds = (label_count * sequences_per_label * per_sequence) + (label_count * warmup_seconds)
    return total_seconds / 60.0


def print_plan(labels: list[str], args: argparse.Namespace) -> None:
    print("=" * 72)
    print("单手手势采集计划")
    print(f"标签总数: {len(labels)}")
    print("标签清单:")
    for i, label in enumerate(labels, start=1):
        print(f"  {i:02d}. {label}")

    eta_min = estimate_collection_minutes(
        label_count=len(labels),
        sequences_per_label=args.sequences_per_label,
        sequence_length=args.sequence_length,
        capture_fps=args.capture_fps,
        warmup_seconds=args.warmup_seconds,
        rest_seconds=args.rest_seconds,
    )

    print("\n建议采集量（单手）:")
    print("  - 最低可用: >= 300 条/标签")
    print("  - 推荐上线: 400~800 条/标签")
    print("  - 高鲁棒版本: 1000 条/标签")
    print("\n本次参数估算:")
    print(f"  - 每标签 {args.sequences_per_label} 条, 每条 {args.sequence_length} 帧")
    print(f"  - 预计总采集时长: {eta_min:.1f} 分钟 (~{eta_min/60:.2f} 小时)")
    print("=" * 72)


def draw_hud(frame, label: str, idx: int, target: int, frame_idx: int, seq_len: int, status: str):
    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Sequence: {idx}/{target}  Frame: {frame_idx}/{seq_len}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)


def main() -> None:
    args = parse_args()
    labels = get_labels(args.labels)
    print_plan(labels, args)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    manifest_path = output_dir / "manifest.csv"
    write_header = not manifest_path.exists()

    detector = MediaPipeDetector()
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 ID={args.camera_id}")

    total_saved = 0
    start_time = time.time()

    with manifest_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["label", "sequence_id", "frames", "feature_dim", "hand_label", "file_path", "timestamp"])

        try:
            for label in labels:
                label_dir = output_dir / label
                ensure_dir(label_dir)

                for remaining in range(int(args.warmup_seconds), 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("摄像头读取失败")
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Prepare '{label}' ... {remaining}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.imshow("Single-Hand Dataset Collector", frame)
                    if cv2.waitKey(1000) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

                collected = 0
                while collected < args.sequences_per_label:
                    sequence = []
                    frame_idx = 0
                    hand_label_of_sequence = "unknown"

                    while frame_idx < args.sequence_length:
                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError("摄像头读取失败")

                        frame = cv2.flip(frame, 1)
                        image, results = detector.detect(frame)
                        detector.draw_landmarks(image, results, use_styled=True)

                        has_active_hand = results.active_hand_landmarks is not None
                        status = f"Active hand: {getattr(results, 'active_hand_label', 'unknown')}"

                        if args.strict_single_hand and not has_active_hand:
                            status = "No valid single hand detected"
                        elif has_active_hand:
                            features = extractor.extract(results)
                            sequence.append(features)
                            frame_idx += 1
                            hand_label_of_sequence = getattr(results, "active_hand_label", "unknown")

                        draw_hud(image, label, collected + 1, args.sequences_per_label, frame_idx, args.sequence_length, status)
                        cv2.imshow("Single-Hand Dataset Collector", image)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt

                    seq_array = np.array(sequence, dtype=np.float32)
                    sequence_id = f"{label}_{int(time.time() * 1000)}_{collected:04d}"
                    file_path = label_dir / f"{sequence_id}.npy"
                    np.save(file_path, seq_array)

                    writer.writerow([
                        label,
                        sequence_id,
                        seq_array.shape[0],
                        seq_array.shape[1],
                        hand_label_of_sequence,
                        str(file_path.as_posix()),
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    ])
                    csvfile.flush()

                    collected += 1
                    total_saved += 1
                    print(f"[{label}] {collected}/{args.sequences_per_label} saved: {file_path.name}")

                    if args.rest_seconds > 0:
                        time.sleep(args.rest_seconds)

        except KeyboardInterrupt:
            print("\n检测到退出，停止采集。")
        finally:
            detector.release()
            cap.release()
            cv2.destroyAllWindows()

    elapsed = (time.time() - start_time) / 60.0
    print("=" * 72)
    print(f"采集结束，总计序列: {total_saved}")
    print(f"实际耗时: {elapsed:.1f} 分钟")
    print(f"Manifest: {manifest_path.resolve()}")
    print("建议: 训练前按 8:1:1 划分 train/val/test 并做标签平衡检查。")
    print("=" * 72)


if __name__ == "__main__":
    main()
