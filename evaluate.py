#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本
运行摔倒检测系统的量化评估

用法：
    python evaluate.py
"""

import sys
import numpy as np

from config import HumanState
from detector import FallDetectionModule
from evaluator import Evaluator
from test_generator import TestSequenceGenerator


def run_evaluation():
    """运行完整评估"""
    gen = TestSequenceGenerator(frame_rate=10, points_per_frame=50)
    detector = FallDetectionModule()
    evaluator = Evaluator()

    # ===== 测试1: 摔倒序列（应检测到） =====
    print("\n[测试1] 摔倒序列 (站立→行走→摔倒→倒地)")
    fall_seq = gen.generate_fall_sequence(noise_level=0.1)
    detector.reset()
    evaluator.reset()

    for state, points in fall_seq:
        avg_h = np.mean(points[:, 2])
        avg_v = np.mean(points[:, 3])
        is_fall, info = detector.detect(avg_h, avg_v, state, points)
        evaluator.add_frame(state, is_fall, info.get('confidence', 0.0))

    result = evaluator.evaluate()
    print(Evaluator.format_report(result))
    assert result.recall > 0.5, f"召回率过低: {result.recall:.2f}"
    print("✓ 摔倒检测召回率合格")

    # ===== 测试2: 站立序列（应无报警） =====
    print("\n[测试2] 站立序列 (应无报警)")
    stand_seq = gen.generate_standing_sequence(num_frames=50, noise_level=0.1)
    detector.reset()
    evaluator.reset()

    for state, points in stand_seq:
        avg_h = np.mean(points[:, 2])
        avg_v = np.mean(points[:, 3])
        is_fall, info = detector.detect(avg_h, avg_v, state, points)
        evaluator.add_frame(state, is_fall, info.get('confidence', 0.0))

    result = evaluator.evaluate()
    print(f"假正例: {result.false_positives}, 误报率: {result.false_alarm_rate:.4f}")
    assert result.false_positives == 0, f"站立序列有 {result.false_positives} 次误报"
    print("✓ 站立序列无误报")

    # ===== 测试3: 行走序列（应无报警） =====
    print("\n[测试3] 行走序列 (应无报警)")
    walk_seq = gen.generate_walking_sequence(num_frames=50, noise_level=0.1)
    detector.reset()
    evaluator.reset()

    for state, points in walk_seq:
        avg_h = np.mean(points[:, 2])
        avg_v = np.mean(points[:, 3])
        is_fall, info = detector.detect(avg_h, avg_v, state, points)
        evaluator.add_frame(state, is_fall, info.get('confidence', 0.0))

    result = evaluator.evaluate()
    print(f"假正例: {result.false_positives}, 误报率: {result.false_alarm_rate:.4f}")
    assert result.false_positives == 0, f"行走序列有 {result.false_positives} 次误报"
    print("✓ 行走序列无误报")

    print("\n" + "=" * 50)
    print("全部测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
