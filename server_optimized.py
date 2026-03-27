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
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict, Optional

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

cfg = Config()

# --------------------------
# GAN模型
# --------------------------
class Generator(nn.Module):
    def __init__(self, attacker_count: int):
        super().__init__()
        self.attacker_count = attacker_count
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, attacker_count * 3), nn.Tanh()
        ).to(DEVICE)

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
# 工具函数
# --------------------------
def lerp_angle(current: float, target: float, speed: float) -> float:
    diff = (target - current + 180) % 360 - 180
    return current + math.copysign(min(abs(diff), speed), diff)

def sphere_to_xyz(radius: float, azimuth: float, elevation: float) -> tuple[float, float, float]:
    x = radius * math.sin(elevation) * math.cos(azimuth)
    y = radius * math.sin(elevation) * math.sin(azimuth)
    z = radius * math.cos(elevation)
    return x, y, z

def compute_threat(ball: dict) -> float:
    dist = ball["dist"]
    speed = ball["speed"]
    if cfg.PRIORITY_MODE == "nearest":
        return dist
    if cfg.PRIORITY_MODE == "fastest":
        return -speed
    return dist * 0.6 - speed * 0.4

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
        for i in range(cfg.ATTACKER_NUM):
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2 - 0.1)  # 严格在上半球
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
        
        self.optD.zero_grad()
        ld = self.bce(self.D(x), y)
        ld.backward()
        self.optD.step()
        
        self.optG.zero_grad()
        lg = -torch.mean(torch.log(self.D(x.detach()) + 1e-7))
        lg.backward()
        self.optG.step()
        
        # 及时释放张量，避免内存泄漏
        self.loss_d_val = float(ld.detach().cpu())
        self.loss_g_val = float(lg.detach().cpu())
        del x, y, ld, lg
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
            self.G = Generator(n)
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
            self._prev_attacker_num = n

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
            if can_launch and len(self.balls) < cfg.MAX_BALLS:
                # 发射后随机漂移到穹顶新位置
                a["ang"] = np.random.uniform(0, 2 * math.pi)
                a["elev"] = np.random.uniform(0.1, math.pi / 2 - 0.1)
            else:
                # 未发射时缓慢移动
                a["ang"] = (a["ang"] + float(g_out[i, 0]) * 0.06 * speed) % (2 * math.pi)
                a["elev"] = max(0.1, min(math.pi / 2 - 0.1, a["elev"] + float(g_out[i, 1]) * 0.04 * speed))
            a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            # 严格限制在上半球，确保z>0
            if a["z"] <= 0:
                a["elev"] = max(0.1, abs(a["elev"]))
                a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            
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
        
        self.balls.sort(key=compute_threat)
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
                    cfg.VIEW_AZ = cmd.get("az", cfg.VIEW_AZ)
                    cfg.VIEW_EL = cmd.get("el", cfg.VIEW_EL)
                    cfg.DOME_SCALE = cmd.get("scale", cfg.DOME_SCALE)
                elif cmd["action"] == "set_priority":
                    cfg.PRIORITY_MODE = cmd.get("mode", "nearest")
                elif cmd["action"] == "update_param":
                    k = cmd.get("key")
                    v = cmd.get("value")
                    if hasattr(cfg, k):
                        if isinstance(getattr(cfg, k), int):
                            v = int(v)
                        elif isinstance(getattr(cfg, k), float):
                            v = float(v)
                        # 参数范围校验
                        if k == "ATTACKER_NUM":
                            v = max(1, min(30, v))
                        elif k == "MAX_BALLS":
                            v = max(20, min(200, v))
                        elif k == "LAUNCH_PER_SECOND":
                            v = max(1, min(30, v))
                        elif k == "LASER_COOLDOWN":
                            v = max(0.05, min(2.0, v))
                        setattr(cfg, k, v)
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
