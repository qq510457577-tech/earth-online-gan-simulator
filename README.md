# 地球 Online GAN 攻防模拟器
## Earth Online GAN Offensive and Defensive Simulator

> **正式全称（毕设/正式交付版）**
> 中文：地球 Online—— 半球型穹顶多攻方饱和攻击 GAN 红外对抗模拟器
> 英文：Earth Online —— Hemispherical Dome Multi-Attacker Saturation Attack GAN Infrared Countermeasure Simulator

---

## 核心标签
`半球型穹顶` `GAN 生成对抗网络` `多目标饱和攻击` `红外对抗` `智能防御` `多目标优先级追踪` `软硬件协同仿真`

*English Tags:* Hemispherical Dome, GAN Generative Adversarial Network, Multi-Target Saturation Attack, Infrared Countermeasure, Intelligent Defense, Multi-Target Priority Tracking, Software-Hardware Collaborative Simulation

---

## 项目简介
### 中文简介
本项目以半球型穹顶为全域空间载体，融合生成对抗网络（GAN）、3D 伪立体可视化、多目标优先级智能追踪与硬件串口控制技术，打造集多攻方随机饱和攻击、单守方红外智能防御、攻防策略自主进化于一体的闭环对抗模拟器。模拟真实全域包围式攻防场景，可用于智能防御仿真、目标追踪算法验证、GAN 对抗学习演示及硬件攻防装置开发，兼具实验研究与教学演示价值。

### English Introduction
Based on the hemispherical dome as the global space carrier, this project integrates Generative Adversarial Network (GAN), 3D pseudo-stereoscopic visualization, multi-target priority intelligent tracking and hardware serial port control technology, creating a closed-loop adversarial simulator integrating multi-attacker random saturation attack, single-defender infrared intelligent defense and autonomous evolution of offensive and defensive strategies. It simulates a real global encircling offensive and defensive scenario, which can be used for intelligent defense simulation, target tracking algorithm verification, GAN adversarial learning demonstration and hardware offensive and defensive device development, with both experimental research and teaching demonstration value.

---

## 核心技术
- 中文：生成对抗网络（GAN）、3D 伪立体可视化、多目标优先级追踪、硬件串口控制、红外激光锁定
- 英文：Generative Adversarial Network (GAN), 3D Pseudo-Stereoscopic Visualization, Multi-Target Priority Tracking, Hardware Serial Port Control, Infrared Laser Locking

---

## 适用场景
- 中文：智能防御仿真、目标追踪算法验证、GAN 对抗学习演示、硬件攻防装置开发、教学实验平台
- 英文：Intelligent Defense Simulation, Target Tracking Algorithm Verification, GAN Adversarial Learning Demonstration, Hardware Offensive and Defensive Device Development, Teaching Experiment Platform

---

## 功能特性

- 🎮 **3D 伪立体渲染** - 半球穹顶 + 经纬线 + 光晕效果
- 🤖 **GAN 自学习** - 攻方策略通过 Generator 动态进化
- 🎯 **智能锁定** - 支持最近/最快/威胁度三种目标优先级
- 📡 **串口控制** - 支持 COM3/9600 波特率云台控制
- 🎨 **动画特效** - 青色彗星尾巴 + 红色爆炸 + 橙色波纹
- 🎛️ **GUI 参数调节** - 13 个滑块实时调节

---

## 运行方法

```bash
pip install pygame torch numpy matplotlib pyserial

python gan_dome_defense.py
```

### 虚拟环境运行（推荐）
```bash
# 创建虚拟环境
python3 -m venv venv
# 激活虚拟环境（Linux/macOS）
source venv/bin/activate
# 激活虚拟环境（Windows）
venv\Scripts\activate
# 安装依赖
pip install pygame torch numpy matplotlib pyserial
# 运行项目
python gan_dome_defense.py
```

---

## 操作说明

| 按键 | 功能 |
|------|------|
| ESC | 退出程序 |
| 鼠标拖动 | 调整视角 |
| 滚轮 | 缩放 |

---

## 参数说明

| 参数 | 说明 |
|------|------|
| AttackerN | 攻击方数量 (10-50) |
| LaunchProb | 发射概率 (0.05-0.6) |
| MaxBalls | 最大小球数 (20-200) |
| BallSpeed | 小球速度 (0.5-6.0) |
| IRRadius | 红外锁定半径 (6-40) |
| GenLR | Generator 学习率 |
| DiscLR | Discriminator 学习率 |
| SimSpeed | 模拟速度倍率 (0.1-3.0x) |

---

## 依赖

- pygame
- torch
- numpy
- matplotlib
- pyserial
