import streamlit as st
import base64
import io
import json
import zipfile
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import requests
from PIL import Image
from websocket import create_connection, WebSocket
from ultralytics import YOLO
import math

# ==============================================
# PP‑YOLOv11 自定义模块（解决c2f_ppblock缺失报错）
# ==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import types

module_name = 'ultralytics.nn.modules.c2f_ppblock'
if module_name not in sys.modules:
    fake_module = types.ModuleType(module_name)
    sys.modules[module_name] = fake_module

class PPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        pool1 = F.adaptive_avg_pool2d(x, 1)
        pool2 = F.adaptive_avg_pool2d(x, 2)
        pool3 = F.adaptive_avg_pool2d(x, 3)
        pool6 = F.adaptive_avg_pool2d(x, 6)

        pool1 = F.interpolate(pool1, size=(H, W), mode='bilinear', align_corners=False)
        pool2 = F.interpolate(pool2, size=(H, W), mode='bilinear', align_corners=False)
        pool3 = F.interpolate(pool3, size=(H, W), mode='bilinear', align_corners=False)
        pool6 = F.interpolate(pool6, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([pool1, pool2, pool3, pool6], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out

class C2f_PPBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.shortcut = shortcut and c1 == c2

        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv1_bn = nn.BatchNorm2d(2 * self.c)
        self.cv1_act = nn.SiLU()

        self.cv2 = nn.Conv2d(self.c, self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.ppblock = PPBlock(self.c, self.c)

        self.cv3 = nn.Conv2d(2 * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv3_bn = nn.BatchNorm2d(c2)
        self.cv3_act = nn.SiLU()

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv1_bn(x)
        x = self.cv1_act(x)
        y1, y2 = x.chunk(2, dim=1)

        y2 = self.cv2(y2)
        y2 = self.ppblock(y2)

        out = torch.cat([y1, y2], dim=1)
        out = self.cv3(out)
        out = self.cv3_bn(out)
        out = self.cv3_act(out)

        if self.shortcut:
            out = out + x
        return out

sys.modules[module_name].C2f_PPBlock = C2f_PPBlock
sys.modules[module_name].PPBlock = PPBlock

def load_model_with_fallback(model_path: str):
    try:
        model = YOLO(model_path, task="detect", verbose=False)
        return model
    except:
        try:
            model = YOLO(model_path, task="detect", verbose=False)
            return model
        except:
            return None

# ==============================================
# 基础依赖
# ==============================================
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

@st.cache_resource(show_spinner=False)
def clear_cache():
    return None
clear_cache()

# ====================== 多语言配置 ======================
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

translations = {
    'zh': {
        'tab_tracking': '📍 轨迹分析',
        'tracking_title': '鱼游动轨迹分析',
        'tracking_upload': '上传视频文件',
        'tracking_run': '🚀 开始轨迹分析',
        'tracking_processing': '正在分析视频轨迹...',
        'total_distance': '总路程（像素）',
        'average_speed': '平均运动速度（像素/秒）',
        'video_duration': '视频时长（秒）',
        'total_frames': '总帧数',
        'health_status': '健康程度',
        'time_period': '分析时间段',
        'daytime': '日间',
        'nighttime': '夜间',
        'no_fish_detected': '未检测到鱼类，无法计算轨迹数据',
        'conf_threshold': '检测置信度阈值',
        'video_process_complete': '视频处理完成！',
        'Ich': '多子小瓜虫病体表病征',
        'Tomont': '多子小瓜虫包囊',
        'Behavior': '金鱼游动行为分析',
        'CiSurface': '刺激隐核虫病体表病症',
        'CiTomont': '刺激隐核虫包囊',
        'CroakerBehavior': '大黄鱼游动行为分析',
        'fuzzy_behavior': '行为特征',
        'fuzzy_surface': '体表特征',
        'fuzzy_pathogen': '病原存在性',
        'healthy': '健康',
        'subhealthy': '亚健康',
        'diseased': '患病',
        'pathogen_absent': '不存在',
        'pathogen_present': '存在',
        'page_title': '鱼类白点病智能检测平台',
        'sidebar_university': '宁波大学 · 水产动物医学综合实验室',
        'sidebar_model': '🧠 模型选择',
        'sidebar_model_type': '模型类型',
        'tab_image': '🖼️ 图片检测',
        'tab_batch': '🗂️ 批量图片',
        'tab_video': '🎞️ 视频检测',
        'tab_camera': '📷 摄像头检测',
        'tab_fuzzy': '🧮 模糊预测',
        'image_upload': '上传图片',
        'image_run': '🚀 开始检测',
        'batch_upload': '选择多张图片',
        'batch_run': '🚀 开始批量检测',
        'video_upload': '上传检测视频',
        'video_run': '🚀 开始视频检测',
        'camera_open': '🎬 打开摄像头',
        'camera_close': '⏹ 关闭摄像头',
        'camera_shot': '点击拍摄',
        'camera_detect': '🔍 开始检测',
        'fuzzy_predict': '🧪 健康预测',
        'fuzzy_result': '风险值: {risk_value} | 状态: {risk_status}',
        'download_traj_video': '下载轨迹视频',
    },
    'en': {
        'tab_tracking': '📍 Trajectory Analysis',
        'tracking_title': 'Fish Trajectory Analysis',
        'tracking_upload': 'Upload Video',
        'tracking_run': '🚀 Analyze',
        'tracking_processing': 'Analyzing...',
        'total_distance': 'Distance',
        'average_speed': 'Speed',
        'video_duration': 'Duration',
        'total_frames': 'Frames',
        'health_status': 'Health',
        'time_period': 'Period',
        'daytime': 'Day',
        'nighttime': 'Night',
        'no_fish_detected': 'No fish detected.',
        'conf_threshold': 'Confidence',
        'video_process_complete': 'Processing Complete!',
        'Ich': 'Ich Surface',
        'Tomont': 'Ich Tomont',
        'Behavior': 'Goldfish Behavior',
        'CiSurface': 'Cryptocaryon Surface',
        'CiTomont': 'Cryptocaryon Tomont',
        'CroakerBehavior': 'Croaker Behavior',
        'fuzzy_behavior': 'Behavior',
        'fuzzy_surface': 'Surface',
        'fuzzy_pathogen': 'Pathogen',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'pathogen_absent': 'Absent',
        'pathogen_present': 'Present',
        'page_title': 'Fish White Spot Detection Platform',
        'sidebar_university': 'Ningbo University · Aquatic Medicine Lab',
        'sidebar_model': '🧠 Model',
        'sidebar_model_type': 'Model Type',
        'tab_image': '🖼️ Image',
        'tab_batch': '🗂️ Batch',
        'tab_video': '🎞️ Video',
        'tab_camera': '📷 Camera',
        'tab_fuzzy': '🧮 Fuzzy',
        'image_upload': 'Upload Image',
        'image_run': '🚀 Detect',
        'batch_upload': 'Upload Images',
        'batch_run': '🚀 Run Batch',
        'video_upload': 'Upload Video',
        'video_run': '🚀 Process Video',
        'camera_open': '🎬 Open Camera',
        'camera_close': '⏹ Close Camera',
        'camera_shot': 'Capture',
        'camera_detect': '🔍 Detect',
        'fuzzy_predict': '🧪 Predict',
        'fuzzy_result': 'Risk: {risk_value} | Status: {risk_status}',
        'download_traj_video': 'Download Video',
    }
}

def t(key):
    return translations[st.session_state.language].get(key, key)

# ====================== 健康程度计算 ======================
def get_health_status(average_speed: float, time_period: str):
    if average_speed > 15:
        return t("healthy")
    elif average_speed >= 5:
        return t("subhealthy")
    else:
        return t("diseased")

# ====================== 页面全局美化 ======================
st.set_page_config(page_title="Fish Detection", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.main { background-color: #f7fcff; }
.top-title-box{
    background: linear-gradient(135deg, #0078ff 0%, #40a9ff 100%);
    padding:28px; border-radius:18px; text-align:center; margin-bottom:24px;
    box-shadow:0 6px 20px rgba(0,120,255,0.15);
}
.top-main-title{ font-size:38px; font-weight:700; color:white; margin:0; letter-spacing:1px; }
.top-sub-title{ font-size:17px; color:#e6f4ff; margin-top:8px; opacity:0.9; }
.stTabs [data-baseweb="tab"] {
    border-radius:10px 10px 0 0; padding:0 18px; font-weight:500;
}
.stButton>button { border-radius:10px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ========== 顶部标题 + 中英文切换 ==========
col1, col2 = st.columns([7, 1])
with col1:
    st.markdown(f"""
    <div class="top-title-box">
        <h1 class="top-main-title">🐟 {t('page_title')}</h1>
        <p class="top-sub-title">Intelligent Detection | Trajectory | Behavior | Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🌐 中文 / EN", use_container_width=True):
        st.session_state.language = "en" if st.session_state.language == "zh" else "zh"
        st.rerun()

# ====================== 模型顺序（严格按你要求） ======================
MODEL_ORDER = [
    ("Ich", "best.pt"),
    ("Tomont", "tomont.best.pt"),
    ("Behavior", "guijibest.pt"),
    ("CiSurface", "cybest.pt"),
    ("CiTomont", "cibest.pt"),
    ("CroakerBehavior", "cyguijibest.pt"),
]

BASE_DIR = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for key, fname in MODEL_ORDER:
        p = BASE_DIR / fname
        if p.exists():
            m = load_model_with_fallback(str(p))
            if m:
                models[key] = m
    return models

MODELS = load_models()
AVAILABLE_MODEL_KEYS = [k for k, _ in MODEL_ORDER if k in MODELS]

# ====================== 工具函数 ======================
def detections_to_df(res):
    rows = []
    if hasattr(res, "boxes") and res.boxes:
        for b in res.boxes:
            rows.append({
                "Class": res.names[int(b.cls[0])],
                "Conf": round(float(b.conf[0]),2),
                "X1Y1X2Y2": [round(float(x),1) for x in b.xyxy[0]]
            })
    return pd.DataFrame(rows)

def predict_on_image(img_bytes, model_key, conf=0.3):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    r = MODELS[model_key].predict(source=img, conf=conf, imgsz=640, verbose=False)[0]
    arr = r.plot()[...,::-1]
    return Image.fromarray(arr), detections_to_df(r)

# ====================== 侧边栏 ======================
with st.sidebar:
    st.markdown(f"### 🎓 {t('sidebar_university')}")
    st.divider()
    st.subheader(t("sidebar_model"))
    model_value = st.selectbox(
        t("sidebar_model_type"),
        options=AVAILABLE_MODEL_KEYS,
        format_func=lambda k: t(k)
    )

# ====================== 标签页 ======================
tabs = st.tabs([
    t('tab_image'), t('tab_batch'), t('tab_video'),
    t('tab_camera'), t('tab_tracking'), t('tab_fuzzy')
])

# 1 图片
with tabs[0]:
    st.subheader(t("tab_image"))
    c1, c2 = st.columns(2)
    with c1:
        f = st.file_uploader(t("image_upload"), type=["jpg","png","jpeg"])
        if f: st.image(f, use_column_width=True)
    with c2:
        if f and st.button(t("image_run"), type="primary", use_container_width=True):
            out_img, df = predict_on_image(f.getvalue(), model_value)
            st.image(out_img, use_column_width=True)
            if not df.empty: st.dataframe(df, use_container_width=True)

# 2 批量
with tabs[1]:
    st.subheader(t("tab_batch"))
    fs = st.file_uploader(t("batch_upload"), accept_multiple_files=True)
    if fs and st.button(t("batch_run"), type="primary"):
        all_df = []
        for f in fs:
            _, df = predict_on_image(f.getvalue(), model_value)
            if not df.empty:
                df["File"] = f.name
                all_df.append(df)
        if all_df:
            st.dataframe(pd.concat(all_df, ignore_index=True), use_container_width=True)

# 3 视频
with tabs[2]:
    st.subheader(t("tab_video"))
    v = st.file_uploader(t("video_upload"), type=["mp4","mov"])
    if v and st.button(t("video_run"), type="primary"):
        tmp = Path("tmp.mp4"); tmp.write_bytes(v.getvalue())
        cap = cv2.VideoCapture(str(tmp))
        ww, hh, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25
        out = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (ww,hh))
        while cap.isOpened():
            ok, fr = cap.read()
            if not ok: break
            r = MODELS[model_value].predict(fr, verbose=False)[0]
            out.write(r.plot())
        cap.release(); out.release()
        st.success(t("video_process_complete"))
        with open("out.mp4","rb") as f:
            st.download_button("📥 Download", f, "result.mp4")

# 4 摄像头
with tabs[3]:
    st.subheader(t("tab_camera"))
    if "cam" not in st.session_state: st.session_state.cam = False
    if not st.session_state.cam:
        if st.button(t("camera_open")): st.session_state.cam = True; st.rerun()
    else:
        if st.button(t("camera_close")): st.session_state.cam = False; st.rerun()
        pic = st.camera_input(t("camera_shot"))
        if pic and st.button(t("camera_detect")):
            im, df = predict_on_image(pic.getvalue(), model_value)
            st.image(im)
            if not df.empty: st.dataframe(df)

# 5 轨迹
with tabs[4]:
    st.subheader("🐠 "+t("tracking_title"))
    v = st.file_uploader(t("tracking_upload"))
    conf = st.slider(t("conf_threshold"), 0.1,1.0,0.3)
    if v and st.button(t("tracking_run"), type="primary"):
        tmp = Path("traj.mp4"); tmp.write_bytes(v.getvalue())
        cap = cv2.VideoCapture(str(tmp))
        ww, hh, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25
        out = cv2.VideoWriter("traj_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (ww,hh))
        cx0, cy0, dist, cnt = None, None, 0, 0
        while cap.isOpened():
            ok, fr = cap.read()
            if not ok: break
            r = MODELS[model_value].predict(fr, conf=conf, verbose=False)[0]
            im = r.plot()
            cx, cy = None, None
            if r.boxes:
                x1,y1,x2,y2 = r.boxes.xyxy[0]
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(im, (cx,cy), 5, (0,0,255), -1)
                if cx0 is not None:
                    cv2.line(im, (cx0,cy0), (cx,cy), (0,0,255), 2)
                    dist += math.hypot(cx-cx0, cy-cy0)
            cx0, cy0 = cx, cy
            out.write(im)
            cnt +=1
        cap.release(); out.release()
        dur = cnt/fps if fps else 0
        spd = dist/dur if dur else 0
        col1,col2,col3=st.columns(3)
        col1.metric(t("total_distance"), round(dist,1))
        col2.metric(t("average_speed"), round(spd,1))
        col3.metric(t("health_status"), get_health_status(spd,""))
        with open("traj_out.mp4","rb") as f:
            st.download_button(t("download_traj_video"), f, "traj.mp4")

# 6 模糊
with tabs[5]:
    st.subheader(t("tab_fuzzy"))
    c1,c2,c3=st.columns(3)
    b = c1.selectbox(t("fuzzy_behavior"), [t("healthy"),t("subhealthy"),t("diseased")])
    s = c2.selectbox(t("fuzzy_surface"), [t("healthy"),t("diseased")])
    p = c3.selectbox(t("fuzzy_pathogen"), [t("pathogen_absent"),t("pathogen_present")])
    if st.button(t("fuzzy_predict"), type="primary"):
        bv = 1 if b==t("healthy") else 2 if b==t("subhealthy") else 3
        sv = 1 if s==t("healthy") else 3
        pv = 1 if p==t("pathogen_absent") else 3
        risk = (bv*0.4 + sv*0.3 + pv*0.3)
        res = t("healthy") if risk<1.7 else t("subhealthy") if risk<2.5 else t("diseased")
        st.success(t("fuzzy_result").format(risk_value=round(risk,1), risk_status=res))
