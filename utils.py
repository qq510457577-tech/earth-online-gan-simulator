# -*- coding: utf-8 -*-
"""
公共工具函数模块
统一管理所有通用工具，消除服务端/本地端重复代码
"""
import math
import numpy as np
from typing import Tuple, Dict

# --------------------------
# 数学工具函数
# --------------------------
def lerp_angle(current: float, target: float, speed: float) -> float:
    """角度线性插值，解决360度环绕问题"""
    diff = (target - current + 180) % 360 - 180
    return current + math.copysign(min(abs(diff), speed), diff)

def sphere_to_xyz(radius: float, azimuth: float, elevation: float) -> Tuple[float, float, float]:
    """球坐标转笛卡尔坐标"""
    x = radius * math.sin(elevation) * math.cos(azimuth)
    y = radius * math.sin(elevation) * math.sin(azimuth)
    z = radius * math.cos(elevation)
    return x, y, z

def gradient_color(start: Tuple[int, int, int], end: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
    """生成渐变颜色"""
    return tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))

def compute_threat(ball: Dict, priority_mode: str) -> float:
    """
    计算目标威胁值，用于优先级排序
    返回值越小优先级越高
    """
    dist = ball["dist"]
    speed = ball["speed"]
    if priority_mode == "nearest":
        return dist
    if priority_mode == "fastest":
        return -speed
    # 威胁模式：距离60% + 速度40%加权
    return dist * 0.6 - speed * 0.4

# --------------------------
# 3D投影相关
# --------------------------
_view_cache = {
    "az": -1.0,
    "el": -1.0,
    "sin_az": 0.0,
    "cos_az": 0.0,
    "sin_el": 0.0,
    "cos_el": 0.0
}

def project(x: float, y: float, z: float, center_x: float, center_y: float, view_az: float, view_el: float, scale: float = 1.0) -> Tuple[int, int]:
    """3D坐标投影到2D屏幕，支持视角旋转，自动缓存三角函数值"""
    global _view_cache
    
    # 仅当视角变化时重新计算三角函数
    if view_az != _view_cache["az"]:
        _view_cache["az"] = view_az
        az = math.radians(view_az)
        _view_cache["sin_az"] = math.sin(az)
        _view_cache["cos_az"] = math.cos(az)
    
    if view_el != _view_cache["el"]:
        _view_cache["el"] = view_el
        el = math.radians(view_el)
        _view_cache["sin_el"] = math.sin(el)
        _view_cache["cos_el"] = math.cos(el)
    
    # 水平旋转（方位角）
    x1 = x * _view_cache["cos_az"] - y * _view_cache["sin_az"]
    y1 = x * _view_cache["sin_az"] + y * _view_cache["cos_az"]
    
    # 垂直旋转（俯仰角）
    y2 = y1 * _view_cache["cos_el"] - z * _view_cache["sin_el"]
    z2 = y1 * _view_cache["sin_el"] + z * _view_cache["cos_el"]
    
    # 等轴投影
    screen_x = int(center_x + y2 * scale)
    screen_y = int(center_y - z2 * scale)
    return screen_x, screen_y

# --------------------------
# 配置校验工具
# --------------------------
PARAM_VALIDATORS = {
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

def validate_param(key: str, value):
    """参数合法性校验，返回合法值"""
    if key not in PARAM_VALIDATORS:
        return value
    try:
        return PARAM_VALIDATORS[key](value)
    except:
        return value
