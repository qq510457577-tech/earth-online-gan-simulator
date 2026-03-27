# -*- coding: utf-8 -*-
"""
地球 Online GAN 攻防模拟器 - Web版
Earth Online GAN Offensive and Defensive Simulator - Web Version
适配pygbag WASM运行环境
"""
import asyncio
import numpy as np
import torch
import torch.nn as nn
import pygame
import math
from collections import deque

# Web环境配置
torch.set_num_threads(1)
DEVICE = torch.device("cpu")  # Web环境只用CPU
HAS_SERIAL = False
HAS_MPL = False  # Web环境禁用matplotlib图表

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

# --------------------------
# 配置类
# --------------------------
class Config:
    WIDTH = 1280
    HEIGHT = 720
    FPS = 60
    BG_COLOR = (5, 8, 25)
    
    HEMI_RADIUS = 260
    DOME_SCALE = 1.0
    DOME_LAYERS = 5
    
    ATTACKER_NUM = 20
    LAUNCH_PROB = 0.2
    MAX_BALLS = 60
    SPEED_BALL = 2.2
    IR_LOCK_RADIUS = 20
    PRIORITY_MODE = "nearest"
    TURRET_SPEED = 6.0
    
    GAN_LR_G = 0.0008
    GAN_LR_D = 0.0015
    GAN_TRAIN_FREQ = 6
    
    SIM_SPEED = 1.0
    BALL_TRAIL = 6
    EXPLOSION_FRAMES = 15
    
    SIM_RUNNING = True
    VIEW_AZ = 30.0
    VIEW_EL = 25.0

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
# 渲染相关
# --------------------------
def project(x: float, y: float, z: float, center_x: float, center_y: float) -> tuple[int, int]:
    scale = cfg.DOME_SCALE
    az = math.radians(cfg.VIEW_AZ)
    el = math.radians(cfg.VIEW_EL)
    
    x1 = x * math.cos(az) - y * math.sin(az)
    y1 = x * math.sin(az) + y * math.cos(az)
    y2 = y1 * math.cos(el) - z * math.sin(el)
    z2 = y1 * math.sin(el) + z * math.cos(el)
    
    screen_x = int(center_x + x1 * scale)
    screen_y = int(center_y - z2 * scale)
    return screen_x, screen_y

class DomeRenderer:
    def __init__(self):
        self._cache = []
        self._build()

    def _build(self):
        R = cfg.HEMI_RADIUS
        self._cache = []
        for layer in range(1, cfg.DOME_LAYERS + 1):
            elev = math.pi / 2 * layer / cfg.DOME_LAYERS
            r_h = R * math.sin(elev)
            z = R * math.cos(elev)
            pts = [(r_h * math.cos(a), r_h * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 72)]
            bright = int(40 + 100 * layer / cfg.DOME_LAYERS)
            self._cache.append(("ring", pts, (20, bright, 120), True))
        for a in np.linspace(0, 2 * math.pi, 18, endpoint=False):
            pts = [sphere_to_xyz(R, a, e) for e in np.linspace(0.02, math.pi / 2, 20)]
            self._cache.append(("merid", pts, (15, 35, 85), False))
        pts = [(R * math.cos(a), R * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 100)]
        self._cache.append(("ground", pts, (20, 80, 200), True))
        for frac in [0.33, 0.66]:
            r2 = R * frac
            pts = [(r2 * math.cos(a), r2 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 60)]
            self._cache.append(("inner", pts, (12, 30, 70), True))

    def draw(self, screen: pygame.Surface, center_x: float, center_y: float):
        for kind, pts, color, closed in self._cache:
            spx = [project(x, y, z, center_x, center_y) for x, y, z in pts]
            pygame.draw.lines(screen, color, closed, spx, 1)
        tx, ty = project(0, 0, cfg.HEMI_RADIUS, center_x, center_y)
        for r, al in [(16, 30), (10, 60), (5, 120)]:
            gs = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (80, 160, 255, al), (r, r), r)
            screen.blit(gs, (tx - r, ty - r))

def compute_threat(ball: dict) -> float:
    dist = ball["dist"]
    speed = ball["speed"]
    if cfg.PRIORITY_MODE == "nearest":
        return dist
    if cfg.PRIORITY_MODE == "fastest":
        return -speed
    return dist * 0.6 - speed * 0.4

# --------------------------
# 主游戏类
# --------------------------
class WebGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("地球 Online GAN 攻防模拟器")
        self.clock = pygame.time.Clock()
        
        # 加载字体
        try:
            self.font = pygame.font.SysFont("simhei, Microsoft YaHei, Arial", 15)
            self.font_b = pygame.font.SysFont("simhei, Microsoft YaHei, Arial", 17, bold=True)
            self.font_s = pygame.font.SysFont("simhei, Microsoft YaHei, Arial", 12)
        except:
            self.font = pygame.font.Font(None, 16)
            self.font_b = pygame.font.Font(None, 18)
            self.font_s = pygame.font.Font(None, 13)

        self.dome = DomeRenderer()
        self.G = Generator(cfg.ATTACKER_NUM)
        self.D = Discriminator()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()
        
        self.attackers = []
        self.balls = []
        self.ball_trails = {}
        self.laser_trails = []
        self.hit_effects = []
        
        self.total_hit = 0
        self.total_launch = 0
        self.frame = 0
        self.loss_g_val = 0.0
        self.loss_d_val = 0.0
        self.turret_pan = 0.0
        self.turret_tilt = 45.0
        self._prev_attacker_num = cfg.ATTACKER_NUM
        
        self._init_attackers()
        self.events = []
        self.log("系统启动 - 模拟器就绪")

    def log(self, msg: str):
        ts = pygame.time.get_ticks() // 1000
        self.events.append(f"[{ts//60:02d}:{ts%60:02d}] {msg}")
        if len(self.events) > 20:
            self.events = self.events[-20:]

    def _init_attackers(self):
        self.attackers = []
        for i in range(cfg.ATTACKER_NUM):
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2)
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
        x = torch.tensor(batch, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        
        self.optD.zero_grad()
        ld = self.bce(self.D(x), y)
        ld.backward()
        self.optD.step()
        
        self.optG.zero_grad()
        lg = -torch.mean(torch.log(self.D(x.detach()) + 1e-7))
        lg.backward()
        self.optG.step()
        
        self.loss_d_val = float(ld)
        self.loss_g_val = float(lg)

    def _update_attackers(self):
        n = cfg.ATTACKER_NUM
        while len(self.attackers) < n:
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2)
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x": x, "y": y, "z": z, "ang": ang, "elev": elev})
        self.attackers = self.attackers[:n]
        
        if n != self._prev_attacker_num:
            self.G = Generator(n)
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
            self._prev_attacker_num = n

        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        noise = torch.randn(1, 10)
        state = torch.tensor([[hit_rate, len(self.balls) / (cfg.MAX_BALLS + 1e-6),
                              (self.frame * 0.001) % 1.0, self.loss_g_val, self.loss_d_val, np.random.rand()]], dtype=torch.float32)
        with torch.no_grad():
            g_out = self.G(torch.cat([noise, state], dim=1)).squeeze(0).numpy()
        
        launched_now = 0
        speed = cfg.SIM_SPEED
        for i, a in enumerate(self.attackers):
            if i >= g_out.shape[0]:
                break
            a["ang"] = (a["ang"] + float(g_out[i, 0]) * 0.06 * speed) % (2 * math.pi)
            a["elev"] = max(0.05, min(math.pi / 2, a["elev"] + float(g_out[i, 1]) * 0.04 * speed))
            a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            
            gan_launch = float(g_out[i, 2])
            launch_threshold = cfg.LAUNCH_PROB * 2 - 1
            gan_decide = gan_launch > launch_threshold
            random_decide = np.random.rand() < cfg.LAUNCH_PROB * 1.5
            can_launch = (gan_decide or (gan_launch < 0.3 and random_decide))
            
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
        
        if launched_now > 0 and self.frame % 60 == 0:
            self.log(f"发射 {launched_now} 个目标，总数 {self.total_launch}")

    def _update_balls(self):
        new_balls = []
        new_trails = {}
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
            bid = id(b)
            if bid not in self.ball_trails:
                self.ball_trails[bid] = []
            trail = self.ball_trails[bid]
            trail.append((b["x"], b["y"], b["z"]))
            if len(trail) > cfg.BALL_TRAIL:
                trail.pop(0)
            new_trails[bid] = trail
            
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
                if in_beam and in_range:
                    self.total_hit += 1
                    hits_now += 1
                    self.hit_effects.append({
                        "x": b["x"], "y": b["y"], "z": b["z"],
                        "life": cfg.EXPLOSION_FRAMES, "max_life": cfg.EXPLOSION_FRAMES
                    })
                    continue
            
            if d > cfg.HEMI_RADIUS + 150:
                continue
            new_balls.append(b)
        
        self.balls = new_balls
        self.ball_trails = new_trails
        if hits_now > 0:
            rate = self.total_hit / (self.total_launch + 1e-6)
            self.log(f"击中 {hits_now} 个！总命中 {self.total_hit} 命中率 {rate:.1%}")

    def _update_turret(self):
        if not self.balls:
            self.laser_trails = [t for t in self.laser_trails if t["life"] > 0]
            for t in self.laser_trails:
                t["life"] -= 1
            return None
        
        self.balls.sort(key=compute_threat)
        target = self.balls[0]
        pan = math.degrees(math.atan2(target["y"], target["x"]))
        tilt = math.degrees(math.atan2(target["z"], math.sqrt(target["x"] ** 2 + target["y"] ** 2) + 1e-6))
        
        speed = cfg.SIM_SPEED
        self.turret_pan = lerp_angle(self.turret_pan, pan, cfg.TURRET_SPEED * speed)
        self.turret_tilt = lerp_angle(self.turret_tilt, tilt, cfg.TURRET_SPEED * speed)
        
        self.laser_trails.append({
            "sx": 0, "sy": 0, "sz": 0,
            "ex": target["x"], "ey": target["y"], "ez": target["z"],
            "life": 15, "max_life": 15
        })
        if len(self.laser_trails) > 20:
            self.laser_trails = self.laser_trails[-15:]
        
        for lt in self.laser_trails:
            lt["life"] -= 1
        self.laser_trails = [lt for lt in self.laser_trails if lt["life"] > 0]
        
        return target

    async def run(self):
        running = True
        mouse_dragging = False
        drag_last = (0, 0)
        
        while running:
            cx = cfg.WIDTH // 2
            cy = cfg.HEIGHT // 2
            self.screen.fill(cfg.BG_COLOR)
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                if e.type == pygame.VIDEORESIZE:
                    cfg.WIDTH, cfg.HEIGHT = e.w, e.h
                    self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT), pygame.RESIZABLE)
                
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    mouse_dragging = True
                    drag_last = e.pos
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    mouse_dragging = False
                if e.type == pygame.MOUSEMOTION and mouse_dragging:
                    dx = e.pos[0] - drag_last[0]
                    dy = e.pos[1] - drag_last[1]
                    cfg.VIEW_AZ = (cfg.VIEW_AZ - dx * 0.4) % 360
                    cfg.VIEW_EL = max(-10.0, min(80.0, cfg.VIEW_EL + dy * 0.3))
                    drag_last = e.pos
                if e.type == pygame.MOUSEWHEEL:
                    cfg.DOME_SCALE = max(0.3, min(3.0, cfg.DOME_SCALE + e.y * 0.05))
            
            # 键盘控制
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                cfg.VIEW_AZ = (cfg.VIEW_AZ + 1.2) % 360
            if keys[pygame.K_RIGHT]:
                cfg.VIEW_AZ = (cfg.VIEW_AZ - 1.2) % 360
            if keys[pygame.K_UP]:
                cfg.VIEW_EL = min(80.0, cfg.VIEW_EL + 0.8)
            if keys[pygame.K_DOWN]:
                cfg.VIEW_EL = max(-10.0, cfg.VIEW_EL - 0.8)
            if keys[pygame.K_SPACE]:
                cfg.SIM_RUNNING = not cfg.SIM_RUNNING
                await asyncio.sleep(0.2)
            
            if cfg.SIM_RUNNING:
                self.frame += 1
                self._update_attackers()
                self._update_balls()
                target = self._update_turret()
                if self.frame % cfg.GAN_TRAIN_FREQ == 0:
                    self._train_gan()
            else:
                target = None
            
            # 渲染
            self.dome.draw(self.screen, cx, cy)
            
            # 激光轨迹
            for lt in self.laser_trails:
                ratio = lt["life"] / lt["max_life"]
                sx, sy = project(lt["sx"], lt["sy"], lt["sz"], cx, cy)
                ex, ey = project(lt["ex"], lt["ey"], lt["ez"], cx, cy)
                color = (int(255 * ratio), int(255 * ratio), 0)
                pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 4)
                pygame.draw.circle(self.screen, (255, 255, 200), (ex, ey), 5)
            
            # 小球和尾巴
            for b in self.balls:
                bx, by, bz = b["x"], b["y"], b["z"]
                sx, sy = project(bx, by, bz, cx, cy)
                bid = id(b)
                if bid in self.ball_trails:
                    trail = self.ball_trails[bid]
                    trail_len = len(trail)
                    for i, (tx, ty, tz) in enumerate(trail):
                        px, py = project(tx, ty, tz, cx, cy)
                        ratio = (i + 1) / trail_len
                        size = max(1, int(6 * ratio))
                        g = int(100 + 155 * ratio)
                        b_col = int(180 + 75 * ratio)
                        pygame.draw.circle(self.screen, (0, g, b_col), (px, py), size)
                pygame.draw.circle(self.screen, (0, 220, 255), (sx, sy), 8)
                pygame.draw.circle(self.screen, (200, 255, 255), (sx, sy), 4)
            
            # 击中特效
            for he in self.hit_effects:
                ratio = he["life"] / he["max_life"]
                hx, hy = project(he["x"], he["y"], he["z"], cx, cy)
                core_radius = int(25 * (1 - ratio) + 3)
                for r in range(core_radius, 0, -2):
                    alpha = int(255 * ratio * (1 - r / (core_radius + 1)))
                    s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(s, (255, 50, 30, alpha), (r + 2, r + 2), r)
                    self.screen.blit(s, (hx - r - 2, hy - r - 2))
                
                ripple_radius = int(35 * (1 - ratio) + 8)
                ripple_alpha = int(180 * ratio)
                s = pygame.Surface((ripple_radius * 2 + 6, ripple_radius * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 150, 0, ripple_alpha), (ripple_radius + 3, ripple_radius + 3), ripple_radius, 2)
                self.screen.blit(s, (hx - ripple_radius - 3, hy - ripple_radius - 3))
                
                pygame.draw.circle(self.screen, (255, 255, 220), (hx, hy), int(5 * ratio) + 1)
            
            self.hit_effects = [he for he in self.hit_effects if he["life"] > 0]
            for he in self.hit_effects:
                he["life"] -= 1
            
            # 防御塔
            scx, scy = project(0, 0, 0, cx, cy)
            pygame.draw.circle(self.screen, (255, 140, 0), (scx, scy), 12)
            pygame.draw.circle(self.screen, (255, 200, 50), (scx, scy), 7)
            pygame.draw.circle(self.screen, (255, 255, 200), (scx, scy), 3)
            
            # 锁定激光
            if target:
                tx, ty = project(target["x"], target["y"], target["z"], cx, cy)
                pygame.draw.line(self.screen, (255, 200, 0), (scx, scy), (tx, ty), 5)
                pygame.draw.line(self.screen, (255, 255, 150), (scx, scy), (tx, ty), 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), 4)
            
            # 信息面板
            info = [
                f"状态: {'运行中' if cfg.SIM_RUNNING else '已暂停'} [空格启停]",
                f"攻击方: {cfg.ATTACKER_NUM}  活跃目标: {len(self.balls)}",
                f"总发射: {self.total_launch}  总命中: {self.total_hit}",
                f"命中率: {self.total_hit / (self.total_launch + 1e-6):.1%}",
                f"模拟速度: {cfg.SIM_SPEED:.1f}x",
                f"视角: 方位{cfg.VIEW_AZ:.0f}° 俯仰{cfg.VIEW_EL:.0f}°",
                f"操作: 拖拽/方向键旋转 | 滚轮缩放 | 空格暂停"
            ]
            for i, t in enumerate(info):
                self.screen.blit(self.font.render(t, True, (200, 220, 255)), (20, 20 + i * 22))
            
            # 日志面板
            log_x = cfg.WIDTH - 300
            log_y = 20
            log_bg = pygame.Surface((280, 180), pygame.SRCALPHA)
            log_bg.fill((8, 14, 42, 200))
            self.screen.blit(log_bg, (log_x, log_y))
            pygame.draw.rect(self.screen, (30, 60, 150), (log_x, log_y, 280, 180), 1)
            self.screen.blit(self.font_b.render("  运行日志", True, (140, 210, 255)), (log_x + 8, log_y + 5))
            for i, line in enumerate(self.events[-8:]):
                color = (100, 255, 130) if "击中" in line else (255, 180, 80) if "发射" in line else (160, 160, 215)
                self.screen.blit(self.font_s.render(line, True, color), (log_x + 8, log_y + 28 + i * 18))
            
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
            await asyncio.sleep(0)
        
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(WebGame().run())
