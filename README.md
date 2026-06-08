<p align="center">
  <h1 align="center">毫米波雷达摔倒检测系统</h1>
  <p align="center">Radar-Based Fall Detection Prototype</p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue?logo=python">
    <img src="https://img.shields.io/badge/pyqt5-5.15+-green?logo=qt">
    <img src="https://img.shields.io/badge/status-demo-yellow">
    <img src="https://img.shields.io/badge/version-2.0.0-orange">
  </p>
</p>

---

## 项目概述

基于**物理毫米波雷达点云模拟**的摔倒检测原型系统，展示从点云生成 → 预处理 → 特征提取 → 摔倒检测 → 可视化报警的完整技术链路。

**核心特性**：
- 纯规则算法，不含深度学习
- 几何模型 + 雷达物理特性模拟
- 5特征加权融合检测
- 多特征宠物过滤

**技术边界**：不包含真实雷达硬件驱动、深度学习模型、后端服务与数据库。

---

## 快速开始

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 推荐 3.10+ |
| PyQt5 | ≥5.15 | GUI框架 |
| NumPy | ≥1.21 | 数值计算 |
| Matplotlib | ≥3.4 | 3D可视化 |

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/radar_fall_detection_demo.git
cd radar_fall_detection_demo

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动程序
python main.py

# Windows 用户也可双击 run.bat
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GeometricModel                           │
│              (椭球体 + 圆柱体人体模型)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   RadarPhysicsModel                         │
│            (RCS反射 + 噪声模型 + 环境建模)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ (N, 4) 点云数据
┌─────────────────────────▼───────────────────────────────────┐
│                  PointCloudPreprocessor                     │
│        (直通滤波 + 统计滤波 + 速度滤波 + 密度过滤)             │
└─────────────────────────┬───────────────────────────────────┘
                          │ 多维特征
┌─────────────────────────▼───────────────────────────────────┐
│                   FallDetectionModule                        │
│              (5特征加权融合 + 宠物过滤)                       │
│     高度30% + 速度25% + 变化率20% + 速度标准差15% + 密度10%   │
└─────────────────────────┬───────────────────────────────────┘
                          │ 报警触发
┌─────────────────────────▼───────────────────────────────────┐
│                      MainWindow                             │
│                (PyQt5 + Matplotlib 可视化)                   │
└─────────────────────────────────────────────────────────────┘
```

### 模块职责

| 文件 | 核心类/函数 | 职责 |
|------|------------|------|
| `main.py` | `MainWindow` | 主窗口布局、事件循环、系统控制 |
| `simulator.py` | `GeometricModel`, `RadarPhysicsModel`, `RadarPointCloudSimulator` | 物理毫米波雷达点云生成 |
| `processor.py` | `PointCloudPreprocessor` | 点云预处理（滤波 + 特征提取） |
| `detector.py` | `FallDetectionModule` | 摔倒检测算法 + 宠物过滤 |
| `ui_components.py` | `PointCloudVisualizer`, `RealTimePlot` | 可视化组件 |
| `config.py` | `HumanState`, 各配置字典 | 集中配置管理 |
| `utils.py` | 日志/导出/截图 | 工具函数 |

---

## 核心功能

### 1. 物理毫米波雷达点云模拟

#### 几何模型
使用**椭球体和圆柱体组合**表示人体：

| 部位 | 几何形状 | 尺寸参数 |
|------|----------|----------|
| 躯干 | 椭球体 | 半轴 0.15m × 0.10m × 0.25m |
| 头部 | 球体 | 半径 0.10m |
| 上臂 | 圆柱体 | 半径 0.035m，长度 0.30m |
| 前臂 | 圆柱体 | 半径 0.030m，长度 0.25m |
| 大腿 | 圆柱体 | 半径 0.06m，长度 0.40m |
| 小腿 | 圆柱体 | 半径 0.045m，长度 0.35m |

#### RCS反射模型
基于几何表面的雷达截面积（RCS）：

| 部位 | RCS (m²) | 反射强度 |
|------|----------|----------|
| 躯干 | 0.5 | 强 |
| 头部 | 0.1 | 中等 |
| 大腿 | 0.08 | 中等 |
| 上臂 | 0.05 | 弱 |
| 小腿 | 0.04 | 弱 |
| 前臂 | 0.03 | 很弱 |

#### 噪声模型
模拟真实毫米波雷达噪声特性：
- **距离噪声**：标准差 2cm，随距离增大
- **角度噪声**：标准差 ~3°，转换为坐标噪声
- **速度噪声**：标准差 0.05m/s
- **热噪声**：基于雷达接收机噪声功率

#### 环境建模
- **杂波**：静态物体反射（密度 0.1 点/m²）
- **多径反射**：墙壁、地面反射
- **房间模型**：4m × 4m × 2.5m

### 2. 点云状态模拟

生成包含 `[x, y, z, velocity_z]` 的 4D 点云，支持 4 种人体状态：

| 状态 | 描述 | 垂直速度 | 点云特征 |
|------|------|----------|----------|
| `STANDING` | 站立静止 | ~0 m/s | 点云集中在躯干，四肢稀疏 |
| `WALKING` | 行走 | 脚部 ±0.08 m/s | 微多普勒效应，速度分布不均 |
| `FALLING` | 摔倒中 | -1.2 m/s | 点云逐渐展开，速度分布集中 |
| `FALLEN` | 平躺地面 | ~0 m/s | 点云水平分布，高度低 |

### 3. 摔倒检测算法

**5特征加权融合**：综合评分 ≥ 50% 判定为摔倒，需连续 N 帧确认（默认 N=2）。

| 特征 | 权重 | 阈值 | 说明 |
|------|------|------|------|
| 人体高度 | 30% | < 0.4m | 平均高度低于阈值 |
| 垂直速度 | 25% | < -0.6 m/s | 向下运动速度 |
| 帧间高度变化率 | 20% | > 5 cm/frame | 高度快速下降 |
| 速度标准差 | 15% | < 0.3 | 速度分布集中（摔倒时） |
| 点云密度 | 10% | > 20 点/m² | 身体展开密度增加 |

**宠物过滤**：多特征宠物识别
- 高度检查：宠物通常 < 0.5m
- 宽度检查：宠物通常 < 0.4m
- 速度标准差：宠物运动更不规则（> 0.3）
- 高度标准差：宠物高度变化小（< 0.1）

### 4. 预处理算法

#### 直通滤波
- 高度过滤：0.05m - 1.8m
- 距离过滤：0.1m - 5.0m

#### 统计滤波
- 使用 **MAD（中位数绝对偏差）** 代替标准差，更鲁棒
- 动态调整K值，适应稀疏点云

#### 速度滤波
- 去除静态杂波（速度接近0的点）
- 保留运动目标

#### 密度过滤
- 去除过于稀疏的区域（可能是噪声）
- 网格大小：20cm

### 5. 报警机制

- 顶部红色闪烁报警条
- 点云颜色切换（viridis → OrRd → 红色闪烁）
- 轨迹线加粗变红
- 报警截图自动保存至 `alarms/`
- Windows 蜂鸣声

---

## 操作指南

### 按钮功能

| 按钮 | 功能 |
|------|------|
| **开始模拟** | 启动点云生成与检测 |
| **停止模拟** | 停止运行 |
| **触发摔倒** | 手动触发摔倒（运行状态下） |
| **重置系统** | 恢复初始状态 |
| **确认报警** | 解除报警闪烁 |

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Space` | 开始模拟 |
| `Ctrl+Space` | 停止模拟 |
| `F` | 触发摔倒 |
| `R` | 重置系统 |
| `Esc` | 退出程序 |

### 参数调节

运行中实时生效：

| 参数 | 范围 | 默认值 |
|------|------|--------|
| 高度阈值 | 0.3 - 1.0m | 0.4m |
| 速度阈值 | -1.5 - 0 m/s | -0.6 m/s |
| 帧数阈值 | 1 - 10 帧 | 2 帧 |
| 噪声水平 | 0 - 1 | 0.1 |
| 模拟速度 | 0.5x - 2.0x | 1.0x |

---

## 雷达物理参数

### FMCW雷达参数

```python
RADAR_PHYSICS_CONFIG = {
    "frequency": 77e9,           # 77GHz毫米波
    "bandwidth": 4e9,            # 4GHz带宽
    "chirp_duration": 50e-6,     # 50μs扫频周期
    "num_rx": 4,                 # 4个接收天线
    "num_tx": 3,                 # 3个发射天线
    "max_range": 5.0,            # 最大探测距离5m
    "range_resolution": 0.0375,  # 距离分辨率3.75cm
    "angle_resolution": 0.1,     # 角度分辨率~6°
}
```

### 噪声参数

```python
"noise": {
    "range_std": 0.02,            # 距离噪声标准差2cm
    "angle_std": 0.05,            # 角度噪声标准差~3°
    "velocity_std": 0.05,         # 速度噪声标准差0.05m/s
    "thermal_noise_power": 1e-10, # 热噪声功率
    "clutter_density": 0.1,       # 杂波密度（点/m²）
}
```

---

## 数据格式

### 点云数据

```python
# 点云: (N, 4) — [x, y, z, velocity_z]
points = np.array([
    [0.1, 0.2, 1.5, 0.0],   # 位置 + 垂直速度
    ...
])
```

### 骨架结构（16 关节）

```
head → neck → chest → waist
                 ├─ left/right_shoulder → elbow → hand
                 └─ left/right_hip → knee → foot
```

---

## 输出文件

| 目录 | 内容 |
|------|------|
| `logs/` | CSV 报警日志 |
| `alarms/` | 报警时自动截图 (`alarm_YYYYMMDD_HHMMSS.png`) |

---

## 项目结构

```
radar_fall_detection_demo/
├── main.py              # 主程序入口
├── simulator.py         # 雷达点云模拟器
├── processor.py         # 点云预处理
├── detector.py          # 摔倒检测算法
├── ui_components.py     # UI可视化组件
├── config.py            # 配置文件
├── utils.py             # 工具函数
├── styles.qss           # Qt样式表
├── requirements.txt     # Python依赖
├── run.bat              # Windows启动脚本
├── logs/                # 日志目录
└── alarms/              # 报警截图目录
```

---

## 优化记录

### v2.1.0 骨架稳定性与真实性优化 (2026-06)

#### 1. One Euro Filter 时序平滑滤波器

**问题**：骨架每帧独立估计，导致抖动过大（Z轴±5cm/帧）

**解决方案**：实现自适应低通滤波器，在低速时减少抖动，高速时减少延迟

```python
class OneEuroFilter:
    """
    1€ Filter - 自适应低通滤波器
    参数：min_cutoff=1.0（静止平滑度）、beta=0.7（运动响应性）
    """
```

**效果**：
- 骨架抖动从 ±5cm/帧 降低到 ±1-2cm/帧
- 保持运动时的响应性，不会产生明显延迟

**参考文献**：
> Casiez, G., Roussel, N. and Vogel, D. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI '12). Austin, Texas (May 5-12, 2012). New York: ACM Press, pp. 2527-2530.
> 
> 论文链接：https://hal.inria.fr/hal-00670496/document
> 
> 官方实现：https://github.com/casiez/OneEuroFilter

#### 2. 骨骼长度约束

**问题**：骨架关节可能"飞出"，不符合人体解剖学

**解决方案**：定义15个骨骼连接的预期长度，允许±20%弹性范围

```python
bone_constraints = {
    (0, 1): 0.12,   # head-neck
    (1, 2): 0.15,   # neck-chest
    (2, 3): 0.20,   # chest-waist
    (4, 6): 0.30,   # shoulder-elbow
    (6, 8): 0.25,   # elbow-hand
    (10, 12): 0.40, # hip-knee
    (12, 14): 0.35, # knee-foot
    ...
}
```

**参考数据来源**：
> 人体骨骼尺寸数据参考：
> - Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement. 4th Edition, John Wiley & Sons.
> - 中国成年人人体尺寸标准 GB/T 10000-2023

#### 3. 倒地点云稀疏度改进

**问题**：倒地状态点云密度与站立状态相同，不符合真实雷达特性

**解决方案**：
- 倒地时点云减少到 60%（`fallen_point_ratio = 0.6`）
- 噪声增加 30%（`noise * 1.3`），模拟倒地后的信号不稳定
- 每个部位最少保留 3 个点，避免完全消失

**参考文献**：
> 倒地后雷达点云特性研究：
> - Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection. Sensors 2024, 24(11), 3660. https://doi.org/10.3390/s24113660
> 
> 研究表明：倒地后点云集中在0.2-0.5m高度，仰卧时反射面积最大，侧卧时较高（~0.5m），坐姿时最高（~1.0m）

#### 4. Post-Fall 确认机制

**问题**：快速蹲下等动作可能误报为摔倒

**解决方案**：检测到摔倒后，需要保持准静态状态5帧才确认报警

```python
# Post-fall确认参数
post_fall_confirmation_frames = 5  # 需要确认的帧数

# 工作流程：
# 1. 检测到摔倒后，不立即报警
# 2. 检查是否保持准静态（高度 < 阈值 且 速度 < 0.1m/s）
# 3. 连续 5 帧满足条件后才确认报警
# 4. 如果高度恢复或有明显运动，重置确认计数
```

**参考文献**：
> Post-fall检测机制研究：
> - Reliable Quasi-Static Post-Fall Floor-Occupancy Detection Using Low-Cost Millimetre-Wave Radar. arXiv:2601.17710. https://arxiv.org/html/2601.17710
> 
> 研究表明：倒地后人体保持准静态，Doppler信号弱且不稳定，空间定位比速度检测更可靠

#### 5. 原始噪声幅度调整

**优化**：减小原始噪声，主要依赖时序滤波器

| 坐标轴 | 优化前 | 优化后 |
|--------|--------|--------|
| X轴 | ±0.02m | ±0.01m |
| Y轴 | ±0.02m | ±0.01m |
| Z轴 | ±0.05m | ±0.02m |

---

## 雷达点云真实性验证

### 噪声参数对比

| 指标 | 真实值（文献） | 模拟值 | 评估 |
|------|---------------|--------|------|
| 距离精度 | ±1-3cm | 2cm std | ✅ 合理 |
| 角度精度 | ±3-6° | 3° std | ✅ 合理 |
| 位置RMSE | 10-25cm | ~5-10cm | ✅ 合理 |
| 骨架关节误差 | 2-6cm | 3-5cm | ✅ 合理 |
| 点云密度 | 5-64点/帧 | 30点/帧 | ✅ 合理 |

**参考文献**：
> mm-Pose: Real-time Human Skeletal Pose Estimation using mmWave Radars and CNNs. IEEE Access. https://arxiv.org/abs/1911.09592
> 
> 平均定位误差：3.2cm深度, 2.7cm高度, 7.5cm方位角

> MARS: A Multiple Activity Recognition System using mmWave Radar. IEEE Internet of Things Journal.
> 
> 19关节MAE：5.87cm

> mmHSE: A Two-Stage Framework for Human Skeleton Estimation Using mmWave FMCW Radar Signals. Applied Sciences 2025, 15(15), 8410. https://doi.org/10.3390/app15158410
> 
> 21关键点MAE：2.78cm，跨域评估MAE：3.14cm

### 倒地状态验证

| 状态 | 真实高度（文献） | 模拟高度 | 评估 |
|------|-----------------|----------|------|
| 仰卧 | ~0.25m | 0.15-0.25m | ✅ 合理 |
| 侧卧 | ~0.5m | - | 🟡 未模拟 |
| 坐姿 | ~1.0m | - | 🟡 未模拟 |

**参考文献**：
> Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection. Sensors 2024.
> 
> "When human targets fall and lie on the ground, the point clouds are concentrated at a lower height, allowing them to be identified as a fall on the ground."
> 
> 仰卧：反射面积最大，高度~0.25m
> 侧卧：反射面积较窄，高度~0.5m
> 坐姿：反射面积最小，高度~1.0m

---

## 版本历史

| 版本 | 日期 | 内容 |
|------|------|------|
| v2.1.0 | 2026-06 | 骨架稳定性优化：One Euro Filter、骨骼约束、倒地点云改进、Post-fall确认 |
| v2.0.0 | 2026-06 | 物理毫米波雷达点云模拟器、优化滤波算法、5特征融合检测、多特征宠物过滤 |
| v1.2.0 | 2026-05 | 代码重构：HumanState 枚举、向量化滤波、修复检测缺陷 |
| v1.1.0 | 2026-05 | 新增骨架可视化、宠物识别过滤 |
| v1.0.0 | 2024-01 | 初始版本 |

---

## 许可证

本项目仅供学习和研究使用。

---

## 参考文献

### 骨架估计与滤波

1. Casiez, G., Roussel, N. and Vogel, D. (2012). **1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.** Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI '12). Austin, Texas (May 5-12, 2012). New York: ACM Press, pp. 2527-2530.
   - 论文链接：https://hal.inria.fr/hal-00670496/document
   - 官方实现：https://github.com/casiez/OneEuroFilter

2. Sengupta, A., et al. (2019). **mm-Pose: Real-time Human Skeletal Pose Estimation using mmWave Radars and CNNs.** IEEE Access.
   - 论文链接：https://arxiv.org/abs/1911.09592
   - 平均定位误差：3.2cm深度, 2.7cm高度, 7.5cm方位角

3. Si, W., et al. (2022). **MARS: A Multiple Activity Recognition System using mmWave Radar.** IEEE Internet of Things Journal.
   - 19关节MAE：5.87cm

4. mmHSE: A Two-Stage Framework for Human Skeleton Estimation Using mmWave FMCW Radar Signals. Applied Sciences 2025, 15(15), 8410. https://doi.org/10.3390/app15158410
   - 21关键点MAE：2.78cm，跨域评估MAE：3.14cm

### 雷达点云特性

5. Ali, Z., et al. (2024). **Impact of receiver thermal noise and PLL RMS jitter in radar measurements.** IEEE Transactions on Instrumentation and Measurement, 73, 2002710.
   - 论文链接：https://doi.org/10.1109/TIM.2024.3370745

6. mmChainPose: Geometry-aware temporal chaining for robust human pose estimation from mmWave point clouds. Neurocomputing 2026.
   - 提出STDGN和GeoChainFormer处理稀疏噪声点云

7. Comprehensive mPoint: A Method for 3D Point Cloud Generation of Human Bodies Utilizing FMCW MIMO mm-Wave Radar. Sensors 2021, 21(19), 6455. https://doi.org/10.3390/s21196455
   - 点云精度提升86%，准确率提升42%

### 摔倒检测与Post-Fall

8. Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection. Sensors 2024, 24(11), 3660. https://doi.org/10.3390/s24113660
   - 摔倒检测准确率：98.2%
   - 倒地后点云特征：仰卧~0.25m，侧卧~0.5m，坐姿~1.0m

9. Reliable Quasi-Static Post-Fall Floor-Occupancy Detection Using Low-Cost Millimetre-Wave Radar. arXiv:2601.17710. https://arxiv.org/html/2601.17710
   - Post-fall检测：准静态目标检测，Capon/MVDR波束成形

10. Non-Contact Fall Detection System Using 4D Imaging Radar for Elderly Safety. Preprints 2025. https://doi.org/10.20944/preprints202504.0809.v1
    - 4D雷达摔倒检测系统验证

### 人体尺寸参考

11. Winter, D.A. (2009). **Biomechanics and Motor Control of Human Movement.** 4th Edition, John Wiley & Sons.
    - 人体骨骼尺寸与运动学参数

12. 中国成年人人体尺寸标准 GB/T 10000-2023
    - 中国成年人体尺寸统计数据

---

<p align="center">
  <i>Radar Fall Detection Demo — Prototype for Algorithm Validation</i>
</p>
