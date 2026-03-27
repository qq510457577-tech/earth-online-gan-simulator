# GAN 半球穹顶防御系统

基于 PyTorch GAN 的 3D 半球穹顶攻防对抗模拟器。

## 功能特性

- 🎮 **3D 伪立体渲染** - 半球穹顶 + 经纬线 + 光晕效果
- 🤖 **GAN 自学习** - 攻方策略通过 Generator 动态进化
- 🎯 **智能锁定** - 支持最近/最快/威胁度三种目标优先级
- 📡 **串口控制** - 支持 COM3/9600 波特率云台控制
- 🎨 **动画特效** - 青色彗星尾巴 + 红色爆炸 + 橙色波纹
- 🎛️ **GUI 参数调节** - 13 个滑块实时调节

## 运行

```bash
pip install pygame torch numpy matplotlib pyserial

python gan_dome_defense.py
```

## 操作说明

| 按键 | 功能 |
|------|------|
| ESC | 退出程序 |
| 鼠标拖动 | 调整视角 |
| 滚轮 | 缩放 |

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

## 依赖

- pygame
- torch
- numpy
- matplotlib
- pyserial
