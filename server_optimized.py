# -*- coding: utf-8 -*-
"""
地球 Online GAN 攻防模拟器 - 优化版后端
修复内存泄漏、提升性能，支持长时间运行
"""
import asyncio
import json
import numpy as np
import torch
import torch.nn as nn
import math
import time
import threading
import gc
import os
import csv
import serial
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Optional
from utils import lerp_angle, sphere_to_xyz, compute_threat, validate_param
import openpyxl

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] 运行设备: {DEVICE}")

class Config:
    HEMI_RADIUS = 260
    ATTACKER_NUM = 15
    LAUNCH_PROB = 0.2
    LAUNCH_PER_SECOND = 10  # 每秒发射小球数量 1-30
    MAX_BALLS = 80
    SPEED_BALL = 2.2
    IR_LOCK_RADIUS = 20
    PRIORITY_MODE = "nearest"
    TURRET_SPEED = 6.0
    LASER_COOLDOWN = 0.2  # 激光冷却时间（秒），控制发射频率，值越小发射越快
    GAN_LR_G = 0.0008
    GAN_LR_D = 0.0015
    GAN_TRAIN_FREQ = 6
    SIM_SPEED = 1.0
    BALL_TRAIL = 15  # 默认拖尾长度加长
    EXPLOSION_FRAMES = 15
    SIM_RUNNING = False
    # 视角参数（可被前端控制）
    VIEW_AZ = 30.0
    VIEW_EL = 25.0
    DOME_SCALE = 1.0
    # 性能优化配置
    GC_INTERVAL = 300  # 每300帧执行一次GC
    PUSH_FPS = 30  # WebSocket推送帧率
    # 串口配置
    SERIAL_PORT = "/dev/ttyUSB0"
    SERIAL_BAUD = 115200
    SERIAL_PARITY = "N"
    SERIAL_STOP_BITS = 1
    SERIAL_TIMEOUT = 1
    # 告警配置
    ALARM_LOW_HIT_RATE_THRESHOLD = 30  # 低于30%
    ALARM_LOW_HIT_RATE_DURATION = 10  # 持续10秒
    ALARM_OVERLOAD_THRESHOLD = 0.8  # 超过MAX_BALLS的80%
    ALARM_LOSS_FLUCTUATION_THRESHOLD = 2.0  # 损失波动超过2倍
    # 配置文件路径
    CONFIG_FILE = "config.json"
    MODEL_SAVE_PATH = "gan_models/"

cfg = Config()

# --------------------------
# GAN模型
# --------------------------
class Generator(nn.Module):
    def __init__(self, attacker_count: int, noise_dim: int = 10, state_dim: int = 6):
        super().__init__()
        self.attacker_count = attacker_count
        self.noise_dim = noise_dim
        self.state_dim = state_dim
        self.input_dim = noise_dim + state_dim
        self.base_layers = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
        ).to(DEVICE)
        self.output_layer = nn.Linear(128, attacker_count * 3).to(DEVICE)
        self.activation = nn.Tanh()

    def update_attacker_count(self, new_count: int) -> None:
        """动态扩展攻击方数量，保留基础层权重，仅重新初始化输出层"""
        self.attacker_count = new_count
        # 仅重新初始化输出层，基础层权重保留
        self.output_layer = nn.Linear(128, new_count * 3).to(DEVICE)
        # 初始化输出层权重
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_layers(x)
        output = self.activation(self.output_layer(features))
        return output.view(-1, self.attacker_count, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.attacker_count, 3)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# --------------------------
# 模拟器核心（优化版）
# --------------------------
class Simulator:
    def __init__(self):
        self.G = Generator(cfg.ATTACKER_NUM)
        self.D = Discriminator()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()
        # 混合精度训练支持
        self.scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
        
        self.attackers: List[Dict] = []
        self.balls: List[Dict] = []
        self.hit_effects: List[Dict] = []
        
        self.total_hit = 0
        self.total_launch = 0
        self.frame = 0
        self.loss_g_val = 0.0
        self.loss_d_val = 0.0
        self.turret_pan = 0.0
        self.turret_tilt = 45.0
        self._prev_attacker_num = cfg.ATTACKER_NUM
        self._prev_lr_g = cfg.GAN_LR_G
        self._prev_lr_d = cfg.GAN_LR_D
        # 发射速率控制
        self._launch_count = 0
        self._last_launch_time = time.time()
        # 激光冷却控制
        self._last_fire_time = 0.0
        # GC控制
        self._gc_counter = 0

        self._init_attackers()
        self.running = True
        self.lock = threading.Lock()
        
        # 启动模拟线程
        threading.Thread(target=self._run_loop, daemon=True).start()
        print("[Simulator] 模拟器启动成功（优化版）")

    def _init_attackers(self):
        self.attackers = []
        min_z = 1  # 攻击点最低z坐标，确保严格在切面上方
        for i in range(cfg.ATTACKER_NUM):
            # 上半球均匀球面采样，真正任意位置随机分布，无密集稀疏区域
            ang = np.random.uniform(0, 2 * math.pi)  # 方位角0~360°均匀随机
            u = np.random.uniform(0.5, 0.99)  # 上半球均匀采样，u∈[0.5,1]对应俯仰角∈[0, π/2 - 0.1]
            elev = math.acos(2 * u - 1)  # 正确的球面均匀采样公式，点分布完全均匀
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x": x, "y": y, "z": z, "ang": ang, "elev": elev})

    def _train_gan(self):
        if len(self.balls) < 4:
            return
        batch, labels = [], []
        for b in self.balls:
            d, sp = b["dist"], b["speed"]
            batch.append([b["x"], b["y"], b["z"], sp, d, 1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
            labels.append([1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
        x = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
        
        # 混合精度训练
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            # 训练判别器
            self.optD.zero_grad()
            d_out = self.D(x)
            ld = self.bce(d_out, y)
        
        self.scaler.scale(ld).backward()
        self.scaler.step(self.optD)
        self.scaler.update()
        
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            # 训练生成器
            self.optG.zero_grad()
            d_out_fake = self.D(x.detach())
            lg = -torch.mean(torch.log(d_out_fake + 1e-7))
        
        self.scaler.scale(lg).backward()
        self.scaler.step(self.optG)
        self.scaler.update()
        
        # 及时释放张量，避免内存泄漏
        self.loss_d_val = float(ld.detach().cpu())
        self.loss_g_val = float(lg.detach().cpu())
        del x, y, ld, lg, d_out, d_out_fake
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    def _update_attackers(self):
        n = cfg.ATTACKER_NUM
        # 实时检查学习率变化，更新优化器
        if self._prev_lr_g != cfg.GAN_LR_G or self._prev_lr_d != cfg.GAN_LR_D:
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
            self._prev_lr_g = cfg.GAN_LR_G
            self._prev_lr_d = cfg.GAN_LR_D
        
        while len(self.attackers) < n:
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2 - 0.1)  # 严格在上半球
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x": x, "y": y, "z": z, "ang": ang, "elev": elev})
        self.attackers = self.attackers[:n]
        
        if n != self._prev_attacker_num:
            # 动态扩展攻击方数量，保留已训练的基础层权重
            self.G.update_attacker_count(n)
            # 优化器重新绑定新的参数
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
            self._prev_attacker_num = n
            print(f"[GAN] 攻击方数量已更新为: {n}，训练权重已保留")

        # 发射速率控制：每秒不超过LAUNCH_PER_SECOND个
        current_time = time.time()
        if current_time - self._last_launch_time >= 1.0:
            self._launch_count = 0
            self._last_launch_time = current_time
        
        # 每秒最大发射数量，模拟速度会影响实际发射速率
        max_launch_this_frame = max(1, int(cfg.LAUNCH_PER_SECOND / 60 * cfg.SIM_SPEED))

        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        noise = torch.randn(1, 10).to(DEVICE)
        state = torch.tensor([[hit_rate, len(self.balls) / (cfg.MAX_BALLS + 1e-6),
                              (self.frame * 0.001) % 1.0, self.loss_g_val, self.loss_d_val, np.random.rand()]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            g_out = self.G(torch.cat([noise, state], dim=1)).squeeze(0).cpu().numpy()
        
        launched_now = 0
        speed = cfg.SIM_SPEED
        for i, a in enumerate(self.attackers):
            # 达到每秒发射上限就不再发射
            if self._launch_count >= cfg.LAUNCH_PER_SECOND or launched_now >= max_launch_this_frame:
                break
            if i >= g_out.shape[0]:
                break

            # 先计算是否发射
            gan_launch = float(g_out[i, 2])
            launch_threshold = cfg.LAUNCH_PROB * 2 - 1
            gan_decide = gan_launch > launch_threshold
            random_decide = np.random.rand() < cfg.LAUNCH_PROB * 1.5
            can_launch = (gan_decide or (gan_launch < 0.3 and random_decide))
            
            # 攻击点随机漂移逻辑：发射后随机漂移到新位置
            min_z = 1  # 攻击点最低z坐标阈值，严格在切面上方
            if can_launch and len(self.balls) < cfg.MAX_BALLS:
                # 发射后随机漂移到穹顶新位置，上半球均匀采样，任意位置随机出现
                a["ang"] = np.random.uniform(0, 2 * math.pi)
                u = np.random.uniform(0.5, 0.99)
                a["elev"] = math.acos(2 * u - 1)
                a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            else:
                # 未发射时缓慢移动，覆盖整个上半球范围
                a["ang"] = (a["ang"] + float(g_out[i, 0]) * 0.06 * speed) % (2 * math.pi)
                a["elev"] = max(0.05, min(math.pi / 2 - 0.05, a["elev"] + float(g_out[i, 1]) * 0.04 * speed))
                a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
                # 移动后强制校验z坐标，低于阈值就重新生成
                if a["z"] < min_z:
                    while True:
                        a["ang"] = np.random.uniform(0, 2 * math.pi)
                        a["elev"] = np.random.uniform(0.05, math.pi / 2 - 0.05)
                        a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
                        if a["z"] >= min_z:
                            break
            
            if len(self.balls) < cfg.MAX_BALLS and can_launch and np.random.rand() < 0.35 * speed:
                speed_val = cfg.SPEED_BALL * (0.8 + np.random.rand() * 0.4)
                R = cfg.HEMI_RADIUS
                base_vx = -a["x"] / R
                base_vy = -a["y"] / R
                base_vz = -a["z"] / R
                
                scatter = math.radians(np.random.uniform(0, 25))
                scatter_az = np.random.uniform(0, 2 * math.pi)
                perp1 = np.array([-base_vy, base_vx, 0.0])
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.array([1.0, 0.0, 0.0])
                else:
                    perp1 = perp1 / np.linalg.norm(perp1)
                base_dir = np.array([base_vx, base_vy, base_vz])
                perp2 = np.cross(base_dir, perp1)
                offset = math.sin(scatter) * (math.cos(scatter_az) * perp1 + math.sin(scatter_az) * perp2)
                final_dir = base_dir * math.cos(scatter) + offset
                norm = np.linalg.norm(final_dir)
                if norm > 1e-6:
                    final_dir /= norm
                vx = float(final_dir[0]) * speed_val
                vy = float(final_dir[1]) * speed_val
                vz = float(final_dir[2]) * speed_val
                
                self.balls.append({
                    "x": a["x"], "y": a["y"], "z": a["z"],
                    "vx": vx, "vy": vy, "vz": vz,
                    "dist": R, "speed": speed_val
                })
                self.total_launch += 1
                launched_now += 1
                self._launch_count += 1
        
        # 及时释放张量
        del noise, state, g_out
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    def _update_balls(self):
        new_balls = []
        hits_now = 0
        speed = cfg.SIM_SPEED
        
        pan_r = math.radians(self.turret_pan)
        tilt_r = math.radians(self.turret_tilt)
        laser_dx = math.cos(tilt_r) * math.cos(pan_r)
        laser_dy = math.cos(tilt_r) * math.sin(pan_r)
        laser_dz = math.sin(tilt_r)
        
        lock_angle_deg = max(2.0, cfg.IR_LOCK_RADIUS * 0.15)
        lock_cos = math.cos(math.radians(lock_angle_deg))
        
        for b in self.balls:
            b["x"] += b["vx"] * speed
            b["y"] += b["vy"] * speed
            b["z"] += b["vz"] * speed
            d = math.sqrt(b["x"] ** 2 + b["y"] ** 2 + b["z"] ** 2)
            sp = math.sqrt(b["vx"] ** 2 + b["vy"] ** 2 + b["vz"] ** 2)
            b["dist"] = d
            b["speed"] = sp
            
            if d > 1e-3:
                bx_n = b["x"] / d
                by_n = b["y"] / d
                bz_n = b["z"] / d
                dot = laser_dx * bx_n + laser_dy * by_n + laser_dz * bz_n
                in_beam = dot > lock_cos
                in_range = d < cfg.HEMI_RADIUS * 0.85
                # 激光冷却判定
                current_time = time.time()
                can_fire = (current_time - self._last_fire_time) >= cfg.LASER_COOLDOWN
                if in_beam and in_range and can_fire:
                    self.total_hit += 1
                    hits_now += 1
                    self._last_fire_time = current_time
                    self.hit_effects.append({
                        "x": b["x"], "y": b["y"], "z": b["z"],
                        "life": cfg.EXPLOSION_FRAMES, "max_life": cfg.EXPLOSION_FRAMES
                    })
                    continue
            
            if d > cfg.HEMI_RADIUS + 150 or b["z"] < 0:  # 限制小球只能在半球水平线（z>0）之上
                continue
            new_balls.append(b)
        
        self.balls = new_balls

    def _update_turret(self):
        if not self.balls:
            return None
        
        self.balls.sort(key=lambda x: compute_threat(x, cfg.PRIORITY_MODE))
        target = self.balls[0]
        pan = math.degrees(math.atan2(target["y"], target["x"]))
        tilt = math.degrees(math.atan2(target["z"], math.sqrt(target["x"] ** 2 + target["y"] ** 2) + 1e-6))
        
        speed = cfg.SIM_SPEED
        self.turret_pan = lerp_angle(self.turret_pan, pan, cfg.TURRET_SPEED * speed)
        self.turret_tilt = lerp_angle(self.turret_tilt, tilt, cfg.TURRET_SPEED * speed)
        
        return target

    def _run_loop(self):
        while self.running:
            start_time = time.time()
            with self.lock:
                if cfg.SIM_RUNNING:
                    self.frame += 1
                    self._update_attackers()
                    self._update_balls()
                    target = self._update_turret()
                    if self.frame % cfg.GAN_TRAIN_FREQ == 0:
                        self._train_gan()
                    
                    # 更新特效
                    self.hit_effects = [he for he in self.hit_effects if he["life"] > 0]
                    for he in self.hit_effects:
                        he["life"] -= 1

                    # 定期GC清理内存
                    self._gc_counter += 1
                    if self._gc_counter >= cfg.GC_INTERVAL:
                        gc.collect()
                        self._gc_counter = 0
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1/60 - elapsed)
            time.sleep(sleep_time)

    def get_state(self) -> dict:
        """获取当前模拟状态，用于前端渲染，仅返回必要数据减少传输量"""
        with self.lock:
            hit_rate = self.total_hit / (self.total_launch + 1e-6)
            return {
                "attackers": [{"x": round(a["x"], 2), "y": round(a["y"], 2), "z": round(a["z"], 2)} for a in self.attackers],
                "balls": [{"x": round(b["x"], 2), "y": round(b["y"], 2), "z": round(b["z"], 2)} for b in self.balls],
                "hits": [{"x": round(he["x"], 2), "y": round(he["y"], 2), "z": round(he["z"], 2), "ratio": round(he["life"]/he["max_life"], 2)} for he in self.hit_effects],
                "turret": {"pan": round(self.turret_pan, 2), "tilt": round(self.turret_tilt, 2)},
                "stats": {
                    "total_launch": self.total_launch,
                    "total_hit": self.total_hit,
                    "hit_rate": round(hit_rate * 100, 1),
                    "ball_count": len(self.balls),
                    "attacker_count": cfg.ATTACKER_NUM,
                    "sim_running": cfg.SIM_RUNNING,
                    "loss_g": round(self.loss_g_val, 6),
                    "loss_d": round(self.loss_d_val, 6),
                    "priority_mode": cfg.PRIORITY_MODE
                }
            }

# --------------------------
# FastAPI服务
# --------------------------
app = FastAPI(title="地球 Online GAN 攻防模拟器")
simulator = Simulator()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/index.html", headers={"Content-Type": "image/x-icon"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] 新客户端连接")
    
    try:
        # 推送状态循环
        while True:
            # 接收前端指令（非阻塞）
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.03)
                cmd = json.loads(data)
                # 处理控制指令
                if cmd["action"] == "toggle_sim":
                    cfg.SIM_RUNNING = not cfg.SIM_RUNNING
                elif cmd["action"] == "update_view":
                    # 视角参数校验
                    if "az" in cmd:
                        try:
                            az = float(cmd["az"])
                            cfg.VIEW_AZ = max(-360.0, min(720.0, az))
                        except (ValueError, TypeError):
                            pass
                    if "el" in cmd:
                        try:
                            el = float(cmd["el"])
                            cfg.VIEW_EL = max(-90.0, min(90.0, el))
                        except (ValueError, TypeError):
                            pass
                    if "scale" in cmd:
                        try:
                            scale = float(cmd["scale"])
                            cfg.DOME_SCALE = max(0.1, min(10.0, scale))
                        except (ValueError, TypeError):
                            pass
                elif cmd["action"] == "set_priority":
                    cfg.PRIORITY_MODE = cmd.get("mode", "nearest")
                elif cmd["action"] == "update_param":
                    k = cmd.get("key")
                    v = cmd.get("value")
                    if not hasattr(cfg, k):
                        continue
                    # 类型转换
                    try:
                        if isinstance(getattr(cfg, k), int):
                            v = int(v)
                        elif isinstance(getattr(cfg, k), float):
                            v = float(v)
                        elif isinstance(getattr(cfg, k), str):
                            v = str(v).strip()
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    # 全参数范围校验
                    param_validators = {
                        "ATTACKER_NUM": lambda x: max(1, min(50, x)),
                        "MAX_BALLS": lambda x: max(20, min(200, x)),
                        "LAUNCH_PROB": lambda x: max(0.0, min(1.0, x)),
                        "LAUNCH_PER_SECOND": lambda x: max(1, min(50, x)),
                        "SPEED_BALL": lambda x: max(0.1, min(20.0, x)),
                        "IR_LOCK_RADIUS": lambda x: max(1, min(90, x)),
                        "TURRET_SPEED": lambda x: max(0.1, min(30.0, x)),
                        "LASER_COOLDOWN": lambda x: max(0.01, min(5.0, x)),
                        "GAN_LR_G": lambda x: max(0.00001, min(0.01, x)),
                        "GAN_LR_D": lambda x: max(0.00001, min(0.01, x)),
                        "GAN_TRAIN_FREQ": lambda x: max(1, min(100, x)),
                        "SIM_SPEED": lambda x: max(0.1, min(10.0, x)),
                        "BALL_TRAIL": lambda x: max(1, min(100, x)),
                        "EXPLOSION_FRAMES": lambda x: max(1, min(100, x)),
                        "HEMI_RADIUS": lambda x: max(50, min(1000, x)),
                        "VIEW_AZ": lambda x: max(-360.0, min(720.0, x)),
                        "VIEW_EL": lambda x: max(-90.0, min(90.0, x)),
                        "DOME_SCALE": lambda x: max(0.1, min(10.0, x)),
                        "PUSH_FPS": lambda x: max(1, min(120, x))
                    }
                    
                    if k in param_validators:
                        v = param_validators[k](v)
                    
                    setattr(cfg, k, v)
                elif cmd["action"] == "reset_sim":
                    # 重置模拟所有数据，保留参数设置
                    cfg.SIM_RUNNING = False
                    with simulator.lock:
                        simulator.total_hit = 0
                        simulator.total_launch = 0
                        simulator.frame = 0
                        simulator.balls = []
                        simulator.hit_effects = []
                        simulator.loss_g_val = 0.0
                        simulator.loss_d_val = 0.0
                        simulator.turret_pan = 0.0
                        simulator.turret_tilt = 45.0
                        simulator._launch_count = 0
                        simulator._last_launch_time = time.time()
                        simulator._last_fire_time = 0.0
                        # 重新初始化攻击者位置
                        simulator._init_attackers()
            except asyncio.TimeoutError:
                pass
            
            # 推送最新状态，30FPS推送，减少带宽和性能开销
            state = simulator.get_state()
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(1/cfg.PUSH_FPS)  # 动态帧率控制
    
    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[WebSocket] 错误: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
