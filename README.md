<p align="center">
  <h1 align="center">毫米波雷达摔倒检测系统</h1>
  <p align="center">Radar-Based Fall Detection Prototype</p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue?logo=python">
    <img src="https://img.shields.io/badge/pyqt5-5.15+-green?logo=qt">
    <img src="https://img.shields.io/badge/status-demo-yellow">
  </p>
</p>

---

## 项目概述

基于**模拟毫米波雷达点云**的摔倒检测原型系统，展示从点云生成→预处理→特征提取→摔倒检测→可视化报警的完整技术链路。纯规则算法，不含深度学习。

**技术边界**：不包含真实雷达硬件驱动、深度学习模型、后端服务与数据库。

---

## 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.8+ |
| PyQt5 | ≥5.15 |
| NumPy | ≥1.21 |
| Matplotlib | ≥3.4 |

### 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
python main.py

# Windows 也可双击 run.bat
```

---

## 系统架构

```
                    ┌─────────────────────┐
                    │    RadarPointCloud   │
                    │     Simulator        │  ← 点云生成
                    │  Standing / Walking  │
                    │  Falling / Fallen    │
                    └──────────┬──────────┘
                               │ (N, 4) point cloud
                    ┌──────────▼──────────┐
                    │   PointCloudPreproc  │  ← 直通滤波 + 统计滤波
                    └──────────┬──────────┘
                               │ avg_height, avg_vz
                    ┌──────────▼──────────┐
                    │  FallDetectionModule │  ← 多特征融合检测
                    │  高度40% + 速度30%    │
                    │  + 高度变化率30%      │
                    └──────────┬──────────┘
                               │ alarm trigger
                    ┌──────────▼──────────┐
                    │      MainWindow     │  ← UI 展示 + 报警
                    │  (PyQt5 + Matplotlib)│
                    └─────────────────────┘
```

### 模块职责

| 文件 | 核心类/函数 | 职责 |
|------|-----------|------|
| `main.py` | `MainWindow` | 主窗口布局、事件循环、系统控制 |
| `simulator.py` | `RadarPointCloudSimulator` | 4D 点云生成，支持 4 种人体状态 |
| `processor.py` | `PointCloudPreprocessor` | 直通滤波 + 统计滤波（向量化） |
| `detector.py` | `FallDetectionModule` | 多特征融合摔倒检测 + 宠物过滤 |
| `ui_components.py` | `PointCloudVisualizer`, `SkeletonVisualizer`, `RealTimePlot`, etc. | 可视化组件 |
| `config.py` | — | 集中配置 + `HumanState` 枚举 |
| `utils.py` | 日志/时间戳/CSV导出/截图/蜂鸣 | 工具函数 |

---

## 核心功能

### 点云模拟

生成包含 `[x, y, z, velocity_z]` 的 4D 点云，基于 16 关节人体骨架模型，模拟毫米波雷达特性（RCS 稀疏性、地面阻力效应、T 分布噪声等）。

| 状态 | 描述 | 垂直速度 |
|------|------|---------|
| `STANDING` | 站立静止 | ~0 m/s |
| `WALKING` | 行走，手臂/腿部摆动 + 身体起伏 | 脚部 ±0.08 m/s |
| `FALLING` | 摔倒中，1.65m → 0.15m，约 1.6s | -1.2 m/s |
| `FALLEN` | 平躺地面 | ~0 m/s |

### 摔倒检测算法

**加权置信度**：综合评分 ≥ 60% 判定为摔倒，需连续 N 帧确认（默认 N=2）。

| 特征 | 权重 | 阈值 |
|------|------|------|
| 人体高度 | 40% | < 0.4m |
| 垂直速度 | 30% | < -0.6 m/s |
| 帧间高度变化率 | 30% | > 5 cm/frame |

**宠物过滤**：低高度 + 低速度物体置信度降低 30%。

### 报警机制

- 顶部红色闪烁报警条
- 点云颜色切换（viridis → OrRd → 红色闪烁）
- 轨迹线加粗变红
- 报警截图自动保存至 `alarms/`
- Windows 蜂鸣声

---

## 操作指南

| 按钮 | 功能 |
|------|------|
| **开始模拟** | 启动点云生成与检测 |
| **停止模拟** | 停止运行 |
| **触发摔倒** | 手动触发摔倒（运行状态下） |
| **重置系统** | 恢复初始状态 |
| **确认报警** | 解除报警闪烁 |

| 快捷键 | 功能 |
|--------|------|
| `Space` | 开始模拟 |
| `Ctrl+Space` | 停止模拟 |
| `F` | 触发摔倒 |
| `R` | 重置系统 |
| `Esc` | 退出程序 |

参数调节（运行中实时生效）：
- **高度阈值**：0.3 - 1.0m（默认 0.4m）
- **速度阈值**：-1.5 - 0 m/s（默认 -0.6 m/s）
- **帧数阈值**：1 - 10 帧（默认 2 帧）
- **噪声水平**：0 - 1（默认 0.1）
- **模拟速度**：0.5x - 2.0x

---

## 数据格式

```python
# 点云: (N, 4) — [x, y, z, velocity_z]
points = np.array([
    [0.1, 0.2, 1.5, 0.0],   # 位置 + 垂直速度
    ...
])
```

骨架结构（16 关节）：
```
head → neck → chest → waist
                 ├─ left/right_shoulder → elbow → hand
                 └─ left/right_hip → knee → foot
```

---

## 输出

| 目录 | 内容 |
|------|------|
| `logs/` | CSV 报警日志 |
| `alarms/` | 报警时自动截图 (`alarm_YYYYMMDD_HHMMSS.png`) |

---

## 版本历史

| 版本 | 内容 | 日期 |
|------|------|------|
| v2.2.0 | 代码重构：HumanState 枚举、向量化滤波、修复检测缺陷 | 2026-05 |
| v2.1.1 | 优化行走状态、修复检测算法 | 2026-05 |
| v2.1.0 | 新增骨架可视化、宠物识别过滤 | 2026-05 |
| v2.0.0 | 初始版本 | 2024-01 |

---

<p align="center">
  <i>Radar Fall Detection Demo — Prototype for Algorithm Validation</i>
</p>
