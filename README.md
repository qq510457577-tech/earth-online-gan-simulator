# 地球 Online GAN 攻防模拟器
## Earth Online GAN Offensive and Defensive Simulator


> 中文：地球 Online—— 半球型穹顶多攻方饱和攻击 GAN 红外对抗模拟器
> 英文：Earth Online —— Hemispherical Dome Multi-Attacker Saturation Attack GAN Infrared Countermeasure Simulator

---

## 核心标签
`半球型穹顶` `GAN 生成对抗网络` `多目标饱和攻击` `红外对抗` `智能防御` `多目标优先级追踪` `软硬件协同仿真`

*English Tags:* Hemispherical Dome, GAN Generative Adversarial Network, Multi-Target Saturation Attack, Infrared Countermeasure, Intelligent Defense, Multi-Target Priority Tracking, Software-Hardware Collaborative Simulation
<img width="1430" height="724" alt="截屏2026-03-27 17 27 51" src="https://github.com/user-attachments/assets/ce10ad1e-8efc-4711-95be-2d81aa8421b9" />
<img width="1424" height="728" alt="截屏2026-03-27 17 30 53" src="https://github.com/user-attachments/assets/5b2f6470-c15a-496a-8059-7599794f1744" />

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

## 硬件原型参考
本仿真项目是以下真实硬件防御系统的数字孪生验证平台，用于算法预研、参数调试、对抗策略训练：

### 一种基于红外探测与高能脉冲激光的穹顶空域高速球体防御系统
#### 技术领域
本发明涉及空域防御技术领域，具体涉及一种针对近球形轻质目标（如乒乓球、高硬塑料球）的高速拦截与毁伤系统，特别适用于 1000 米以内穹顶式全向防御场景。

#### 背景技术（现有问题）
目前针对低空慢速下落球体的拦截，多采用机械弹射（弹簧、摩擦轮）或实弹射击方式：
- **机械方式**：发射频率受物理惯性限制，最高仅能达到 50Hz 左右，无法应对每秒数百次的高速目标流。
- **传统火力**：成本高昂、附带损伤大，且弹道有延迟，对于 300 米距离的微小目标，瞄准反应时间不足，无法实现精准击穿。
- **核心痛点**：现有技术均无法在 300 米距离上，以超过 500 次 / 秒的频率对空中移动目标实现实时精准击穿。

#### 发明内容（核心方案）
本发明提供一种基于红外主动探测与近红外高能脉冲激光结合的非机械防御系统。通过红外视觉实时锁定目标，FPGA 高速计算生成瞄准指令，控制高速振镜引导激光束瞬间烧穿目标，实现 **500Hz+** 的超高频拦截。

##### 系统整体架构
系统由红外探测子系统、智能控制子系统、激光指向子系统及高能毁伤子系统四部分组成。

##### 关键部件与参数（硬性指标）
1. **红外探测子系统（眼睛）**
   - 体制：主动红外点阵照明 + 高速红外相机阵列。
   - 型号建议：Basler acA2040-180km (NIR 版本) 或 同等级全局快门相机。
   - 参数：
     - 帧率：≥1000 帧 / 秒 (fps)。
     - 分辨率：2040×1080 及以上。
     - 波段：850nm - 940nm 近红外波段（对人眼不可见，对塑料反射强）。
     - 探测距离：1000 米处光斑识别率 ≥95%。

2. **智能控制子系统（大脑）**
   - 核心：FPGA + 嵌入式 AI 加速模块。
   - 选型建议：Xilinx Kria K26 SOM (Zynq UltraScale+ MPSoC)。
   - 性能：
     - 解算延迟：≤500 微秒 (μs)。
     - 目标跟踪容量：同时锁定 ≥100 个目标。
     - 算法：基于 YOLO 的微型目标检测算法 + 卡尔曼滤波预测。

3. **激光指向子系统（舵手）**
   - 核心：高速双轴振镜扫描仪 (Galvo Scanner)。
   - 选型建议：军工级振镜（需定制低延迟版），参考型号：GSI Group 62xx 系列或国产同级。
   - 参数：
     - 响应频率：≥20kHz。
     - 扫描角度：±20° 及以上。
     - 定位精度：≤0.1 mrad。
     - 切换时间：≤1ms。

4. **高能毁伤子系统（武器）**
   - 核心：近红外脉冲激光器。
   - 参数（最关键的杀伤链路）：
     - 波长：1064nm (对高硬塑料烧蚀效率最高)。
     - 单脉冲能量：50mJ - 80mJ。
     - 脉冲宽度：10ns - 20ns (高峰值功率)。
     - 重复频率：1000Hz (满足 500Hz+ 设计冗余)。
     - 光束质量：M² < 1.3。

#### 权利要求书（核心保护点）
1. 一种穹顶防御系统，其特征在于，包括：一个红外探测单元，用于获取空域内目标的三维位置信息；一个控制处理单元，用于在小于 1 毫秒的时间内计算出目标瞄准点；一个激光发射单元，包含高能脉冲激光器和高速振镜，用于根据所述瞄准点发射激光束；所述激光束的脉冲频率不低于 1000Hz，且能够在 1000 米距离内聚焦于直径小于 4 厘米的物体表面。
2. 根据权利要求 1 所述的系统，其特征在于，所述高能脉冲激光器的波长为 1064nm，单脉冲能量范围为 50-80mJ，脉冲宽度为 10-20ns。
3. 根据权利要求 1 所述的系统，其特征在于，所述红外探测单元的帧率不低于 1000 帧 / 秒，采用近红外波段主动照明。

#### 具体实施例
工作流程：
1. **探测**：红外相机以 1000fps 实时拍摄下落的乒乓球。
2. **识别**：FPGA 实时识别目标坐标，计算出当前速度和下一刻预测位置。
3. **指向**：控制单元计算出振镜需要偏转的角度，指令发送至振镜驱动器。
4. **击穿**：振镜快速将激光束引导至目标重心位置。50mJ 能量的激光在 10ns 内瞬间烧穿乒乓球壳，造成内部气压失衡或轨迹破坏，实现防御。

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
- fastapi
- uvicorn
- websockets

---

## 完整参数说明

| 参数名称 | 调节范围 | 功能说明 | 状态 |
|---------|---------|---------|------|
| 攻击方数量 | 1~30个 | 后端实时更新攻击点数量，GAN模型自动适配 | ✅ 正常 |
| 发射概率 | 0.05~0.6 | 控制单个攻击点发射意愿，影响发射随机性 | ✅ 正常 |
| 每秒发射数量 | 1~30个/秒 | 全局发射上限，精准控制弹幕密度 | ✅ 正常 |
| 最大小球数量 | 20~200个 | 限制同时存在的小球总数，避免卡顿 | ✅ 正常 |
| 小球速度 | 0.5~6.0单位/帧 | 控制小球飞行速度 | ✅ 正常 |
| 红外锁定半径 | 6~40单位 | 控制激光锁定范围，越大越容易命中 | ✅ 正常 |
| 云台速度 | 1.0~15.0°/帧 | 控制炮塔旋转速度，越快跟瞄越准 | ✅ 正常 |
| 激光冷却时间 | 0.05~2.0秒 | 控制激光发射频率，越小发射越快，攻防难度越低 | ✅ 正常 |
| 模拟速度倍率 | 0.1~3.0× | 全局速度控制，支持慢放/加速 | ✅ 正常 |
| 半球穹顶半径 | 150~350单位 | 实时调整穹顶大小，前端自动重建网格 | ✅ 正常 |
| 小球尾巴长度 | 0~20帧 | 前端本地控制拖尾长度，调节即时生效 | ✅ 正常 |
| 爆炸效果帧数 | 5~30帧 | 控制爆炸动画持续时间 | ✅ 正常 |
| 生成器学习率 | 0.0001~0.005 | 实时更新GAN优化器参数，进化速度可调 | ✅ 正常 |
| 判别器学习率 | 0.0001~0.005 | 同上 | ✅ 正常 |
| 优先级模式 | 最近/最快/威胁 | 点击切换即时生效，锁定逻辑自动切换 | ✅ 正常 |

---

## 部署说明

### Web版部署
1. 安装依赖：`pip install -r requirements.txt`
2. 启动服务：`python server.py`
3. 访问地址：`http://服务器IP:8501`

### 桌面版运行
直接运行：`python gan_dome_defense.py`
