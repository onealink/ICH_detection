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
# 🔥 嵌入你的 PP-YOLOv11 自定义模块（解决报错）
# ==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f

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
            out = out + x[:, :self.c * 2] if x.shape[1] == self.c * 2 else out + x
        return out


# 安全加载自定义模型（自动兼容 PP-YOLOv11）
def safe_load_yolo(model_path):
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([C2f_PPBlock, PPBlock])
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败（已包含PP-YOLOv11模块）: {str(e)}")

# ==============================================
# 以下是你原来的完整代码，100% 不变！
# ==============================================

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

# ====================== 正确清除缓存 ======================
@st.cache_resource(show_spinner=False)
def clear_cache():
    return None
clear_cache()

# ====================== 语言配置 ======================
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
        'model_loaded': '成功加载模型：{k} -> {p}',
        'model_not_found': '模型文件不存在：{p}（{k}模型）',
        'model_switch': '模型{k}不存在，已切换为{default_model}',
        'no_available_model': '无可用模型，请检查模型文件路径！',
        'fallback_model': '所有模型加载失败，已兜底加载Ich模型',
        'conf_threshold': '检测置信度阈值（降低以检测更多目标）',
        'conf_threshold_help': '阈值越低，检测到的目标越多（可能包含误检）',
        'max_frames': '最大分析帧数（0=无限制）',
        'max_frames_help': '设置最大分析帧数，0表示分析全部帧',
        'original_video': '原始视频',
        'analysis_results': '分析结果',
        'processed_video_download': '处理后视频下载',
        'download_traj_video': '下载带轨迹的检测视频',
        'video_format_help': '支持常见视频格式，建议时长不超过1分钟以保证分析速度',
        'cannot_read_video': '无法读取视频',
        'cannot_read_video_file': '无法读取视频文件',
        'frame_inference_failed': '帧推理失败:',
        'analysis_failed': '分析失败：',
        'video_process_complete': '视频处理完成！',
        'suggestions': '建议：1.降低置信度阈值 2.确认视频中有鱼类 3.检查模型类别是否匹配',
        'model_label': '模型：',
        'health_status_label': '健康程度：',
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
        'page_title': 'YOLO病害检测',
        'header_title': '鱼类寄生虫病检测',
        'header_subtitle': '图片 / 批量 / 视频 / 摄像头 / 轨迹分析 / 模糊预测 — 一站式检测台',
        'sidebar_university': '宁波大学 \n 水产动物医学综合实验室',
        'sidebar_model': '🧠 模型与参数',
        'sidebar_model_type': '模型类型',
        'sidebar_current_model': '当前模型:',
        'tab_image': '🖼️ 图片检测',
        'tab_batch': '🗂️ 批量图片',
        'tab_video': '🎞️ 视频检测',
        'tab_camera': '📷 摄像头检测',
        'tab_fuzzy': '🧮 模糊预测',
        'image_original': '原图',
        'image_detection': '检测与结果',
        'image_upload': '上传图片',
        'image_run': '🚀 开始检测',
        'image_result': '检测结果',
        'image_download_excel': '下载 Excel（检测表）',
        'image_download_img': '下载 标注图片',
        'batch_upload': '选择多张图片',
        'batch_run': '🚀 开始批量检测',
        'batch_processing': '推理中：',
        'batch_total': '总数：',
        'batch_no_results': '未检测到目标。',
        'batch_download_excel': '📥 下载 Excel（批量检测表）',
        'batch_download_zip': '🗜️ 打包下载 标注图片ZIP',
        'video_upload': '上传检测视频',
        'video_run': '🚀 开始视频检测',
        'video_disabled': '当前云端环境未能加载 OpenCV（cv2），视频处理功能已禁用。请在本地运行或安装支持的 OpenCV 版本。',
        'video_processing': '本地视频处理...（按 CPU 速度可能较慢）',
        'video_download': '下载处理后视频',
        'camera_title': '📷 摄像头检测（拍照版）',
        'camera_caption': '点击“打开摄像头”后才渲染拍照控件；点击“关闭摄像头”停止并隐藏。',
        'camera_open': '🎬 打开摄像头',
        'camera_close': '⏹ 关闭摄像头',
        'camera_not_started': '摄像头未开启。点击“打开摄像头”开始拍照。',
        'camera_shot': '点击下方按钮拍一张',
        'camera_detect': '检测此照片',
        'fuzzy_title': '🧮 模糊预测',
        'fuzzy_input': '输入指标参数',
        'fuzzy_predict': '🧪 预测',
        'fuzzy_result': '风险值: {risk_value}，状态: {risk_status}',
        'category': '类别',
        'confidence': '置信度',
        'location': '位置',
        'path': '路径',
        'fuzzy_calc_error': '模糊计算异常，已使用默认值：'
    },
    'en': {
        'tab_tracking': '📍 Trajectory Analysis',
        'tracking_title': 'Fish Swimming Trajectory Analysis',
        'tracking_upload': 'Upload Video File',
        'tracking_run': '🚀 Start Trajectory Analysis',
        'tracking_processing': 'Analyzing video trajectory...',
        'total_distance': 'Total Distance (pixels)',
        'average_speed': 'Average Movement Speed (pixels/sec)',
        'video_duration': 'Video Duration (sec)',
        'total_frames': 'Total Frames',
        'health_status': 'Health Status',
        'time_period': 'Analysis Time Period',
        'daytime': 'Daytime',
        'nighttime': 'Nighttime',
        'no_fish_detected': 'No fish detected, cannot calculate trajectory data',
        'model_loaded': 'Successfully loaded model: {k} -> {p}',
        'model_not_found': 'Model file not found: {p} ({k} model)',
        'model_switch': 'Model {k} does not exist, switched to {default_model}',
        'no_available_model': 'No available models, please check model file path!',
        'fallback_model': 'All models failed to load, fallback to Ich model',
        'conf_threshold': 'Detection Confidence Threshold (lower to detect more targets)',
        'conf_threshold_help': 'Lower threshold detects more targets (may include false detections)',
        'max_frames': 'Maximum Analysis Frames (0=unlimited)',
        'max_frames_help': 'Set maximum frames to analyze, 0 means analyze all frames',
        'original_video': 'Original Video',
        'analysis_results': 'Analysis Results',
        'processed_video_download': 'Processed Video Download',
        'download_traj_video': 'Download video with trajectory detection',
        'video_format_help': 'Supports common video formats, recommended duration ≤ 1 minute for speed',
        'cannot_read_video': 'Cannot read video',
        'cannot_read_video_file': 'Cannot read video file',
        'frame_inference_failed': 'Frame inference failed:',
        'analysis_failed': 'Analysis failed: ',
        'video_process_complete': 'Video processing completed!',
        'suggestions': 'Suggestions: 1.Lower confidence threshold 2.Confirm video contains fish 3.Check model category matching',
        'model_label': 'Model: ',
        'health_status_label': 'Health Status: ',
        'Ich': 'Ichthyophthirius Surface Symptoms',
        'Tomont': 'Ichthyophthirius Tomont',
        'Behavior': 'Goldfish Swimming Behavior Analysis',
        'CiSurface': 'Cryptocaryon irritans Surface Symptoms',
        'CiTomont': 'Cryptocaryon irritans Tomont',
        'CroakerBehavior': 'Large Yellow Croaker Swimming Behavior Analysis',
        'fuzzy_behavior': 'Behavior Feature',
        'fuzzy_surface': 'Surface Features',
        'fuzzy_pathogen': 'Pathogen Existence',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'pathogen_absent': 'Absent',
        'pathogen_present': 'Present',
        'page_title': 'YOLO Disease Detection',
        'header_title': 'Fish Parasitic Disease Detection',
        'header_subtitle': 'Image / Batch / Video / Camera / Trajectory Analysis / Fuzzy Prediction — One-stop Detection Platform',
        'sidebar_university': 'Ningbo University \n Aquatic Animal Medicine Laboratory',
        'sidebar_model': '🧠 Model & Parameters',
        'sidebar_model_type': 'Model Type',
        'sidebar_current_model': 'Current Model:',
        'tab_image': '🖼️ Image Detection',
        'tab_batch': '🗂️ Batch Images',
        'tab_video': '🎞️ Video Detection',
        'tab_camera': '📷 Camera Detection',
        'tab_fuzzy': '🧮 Fuzzy Prediction',
        'image_original': 'Original Image',
        'image_detection': 'Detection & Results',
        'image_upload': 'Upload Image',
        'image_run': '🚀 Start Detection',
        'image_result': 'Detection Result',
        'image_download_excel': 'Download Excel (Detection Table)',
        'image_download_img': 'Download Annotated Image',
        'batch_upload': 'Select Multiple Images',
        'batch_run': '🚀 Start Batch Detection',
        'batch_processing': 'Processing: ',
        'batch_total': 'Total: ',
        'batch_no_results': 'No targets detected.',
        'batch_download_excel': '📥 Download Excel (Batch Detection)',
        'batch_download_zip': '🗜️ Download Annotated Images (ZIP)',
        'video_upload': 'Upload Video',
        'video_run': '🚀 Start Video Detection',
        'video_disabled': 'OpenCV (cv2) not loaded in current cloud environment. Video processing disabled. Please run locally or install supported OpenCV version.',
        'video_processing': 'Local video processing... (may be slow depending on CPU)',
        'video_download': 'Download Processed Video',
        'camera_title': '📷 Camera Detection (Photo Mode)',
        'camera_caption': 'Camera widget loads only after clicking "Open Camera"; click "Close Camera" to stop and hide.',
        'camera_open': '🎬 Open Camera',
        'camera_close': '⏹ Close Camera',
        'camera_not_started': 'Camera not started. Click "Open Camera" to begin.',
        'camera_shot': 'Click button below to take photo',
        'camera_detect': 'Detect This Photo',
        'fuzzy_title': '🧮 Fuzzy Prediction',
        'fuzzy_input': 'Input Indicator Parameters',
        'fuzzy_predict': '🧪 Predict',
        'fuzzy_result': 'Risk Value: {risk_value}, Status: {risk_status}',
        'category': 'Category',
        'confidence': 'Confidence',
        'location': 'Location',
        'path': 'Path',
        'fuzzy_calc_error': 'Fuzzy calculation error, using default value: '
    }
}

def t(key):
    return translations[st.session_state.language].get(key, key)

# ====================== 健康程度计算 ======================
def get_health_status(average_speed: float, time_period: str) -> str:
    is_daytime = time_period in [t("daytime"), "日间", "Daytime"]
    if st.session_state.language == 'zh':
        healthy = "健康"
        subhealthy = "亚健康"
        diseased = "患病"
    else:
        healthy = t("healthy")
        subhealthy = t("subhealthy")
        diseased = t("diseased")

    if is_daytime:
        if average_speed > 15:
            return healthy
        elif 10 <= average_speed <= 15:
            return subhealthy
        else:
            return diseased
    else:
        if average_speed > 10:
            return healthy
        elif 5 <= average_speed <= 10:
            return subhealthy
        else:
            return diseased

# ====================== 页面配置 ======================
st.set_page_config(page_title=t('page_title'), page_icon="🧪", layout="wide")

# ====================== 模型加载（已支持 PP-YOLOv11） ======================
BASE_DIR = Path(__file__).parent

WEIGHTS = BASE_DIR / "best.pt"
TOMONT_WEIGHTS = BASE_DIR / "tomont.best.pt"
BEHAVIOR_WEIGHTS = BASE_DIR / "guijibest.pt"
CI_SURFACE_WEIGHTS = BASE_DIR / "cybest.pt"
CI_TOMONT_WEIGHTS = BASE_DIR / "cibest.pt"
CROAKER_BEHAVIOR_WEIGHTS = BASE_DIR / "cyguijibest.pt"

MODEL_PATHS = {
    "Ich": str(WEIGHTS),
    "Tomont": str(TOMONT_WEIGHTS),
    "Behavior": str(BEHAVIOR_WEIGHTS),
    "CiSurface": str(CI_SURFACE_WEIGHTS),
    "CiTomont": str(CI_TOMONT_WEIGHTS),
    "CroakerBehavior": str(CROAKER_BEHAVIOR_WEIGHTS),
}
DEFAULT_CONF = 0.6

@st.cache_resource(show_spinner=True)
def load_models():
    models = {}
    st.write("🔍 模型目录：", BASE_DIR)
    for k, p in MODEL_PATHS.items():
        path = Path(p)
        if path.exists():
            try:
                models[k] = safe_load_yolo(str(path))
                st.success(f"✅ {k} 加载成功")
            except Exception as e:
                st.error(f"❌ {k} 失败：{str(e)[:100]}")
        else:
            st.warning(f"⚠️ {k} 不存在")
    return models

MODELS = load_models()

# ====================== 工具函数（完全不变） ======================
def detections_to_df(res) -> pd.DataFrame:
    if hasattr(res, "boxes") and hasattr(res, "names"):
        rows = []
        names = getattr(res, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_np = boxes.cls.detach().cpu().numpy().astype(int)
            conf_np = boxes.conf.detach().cpu().numpy()
            xyxy_np = boxes.xyxy.detach().cpu().numpy()
            for i in range(len(cls_np)):
                rows.append({t("category"): names.get(int(cls_np[i]), str(int(cls_np[i]))), t("confidence"): float(conf_np[i]), t("location"): [float(x) for x in xyxy_np[i].tolist()]})
        return pd.DataFrame(rows)
    if isinstance(res, list):
        rows = []
        for d in res or []:
            rows.append({t("category"): d.get("category") or d.get("class_name") or d.get("name") or d.get("cls"), t("confidence"): d.get("conf") or d.get("confidence"), t("location"): d.get("location") or d.get("bbox") or d.get("xyxy"), t("path"): d.get("path")})
        return pd.DataFrame(rows)
    if isinstance(res, pd.DataFrame):
        return res
    return pd.DataFrame()

def predict_on_image(img_input, model_key: str, conf: float | None = None):
    if isinstance(img_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, Image.Image):
        pil_img = img_input.convert("RGB")
    elif isinstance(img_input, (str, Path)):
        pil_img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:
            pil_img = Image.fromarray(img_input)
        elif img_input.ndim == 3:
            if CV2_OK:
                pil_img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(img_input[..., ::-1])
        else:
            raise TypeError(f"Unsupported numpy shape: {img_input.shape}")
    else:
        raise TypeError(f"Unsupported type: {type(img_input)}")
    c = float(conf) if conf is not None else DEFAULT_CONF
    if model_key not in MODELS:
        default_model = list(MODELS.keys())[0] if MODELS else None
        model_key = "Ich" if "Ich" in MODELS else default_model
    if not model_key:
        raise RuntimeError(t('no_available_model'))
    r = MODELS[model_key].predict(source=pil_img, conf=c, imgsz=640, verbose=False)[0]
    im_bgr = r.plot()
    im_rgb = im_bgr[..., ::-1]
    vis_pil = Image.fromarray(im_rgb)
    df = detections_to_df(r)
    return vis_pil, df

def process_video(video_bytes: bytes, model_key: str, conf: float | None = None, max_frames: int | None = None) -> Path:
    if not CV2_OK:
        raise RuntimeError(t("video_disabled"))
    in_path = Path("input_tmp.mp4")
    in_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(t("cannot_read_video"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(f"processed_{int(time.time())}.mp4")
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if max_frames and i > max_frames:
            break
        c = float(conf) if conf is not None else DEFAULT_CONF
        if model_key not in MODELS:
            default_model = list(MODELS.keys())[0] if MODELS else None
            model_key = "Ich" if "Ich" in MODELS else default_model
        if not model_key:
            raise RuntimeError(t('no_available_model'))
        r = MODELS[model_key].predict(source=frame, conf=c, imgsz=640, verbose=False)[0]
        vw.write(r.plot())
    cap.release()
    vw.release()
    return out_path

def calculate_fish_trajectory(video_bytes: bytes, model_key: str, conf: float = DEFAULT_CONF, max_frames: int = None) -> dict:
    if not CV2_OK:
        return {"success": False, "message": t("video_disabled"), "total_distance": 0, "average_speed": 0, "video_duration": 0, "total_frames": 0, "processed_video_path": ""}
    if model_key not in MODELS:
        default_model = list(MODELS.keys())[0] if MODELS else None
        model_key = default_model
    if not model_key:
        return {"success": False, "message": t('no_available_model'), "total_distance": 0, "average_speed": 0, "video_duration": 0, "total_frames": 0, "processed_video_path": ""}
    prev_center = None
    total_distance = 0.0
    total_frames = 0
    trajectory_points = []
    current_model = MODELS[model_key]
    model_class_names = current_model.names
    fish_keywords = ["healthy", "subhealthy", "diseased", "健康", "亚健康", "患病", "鱼", "fish"]
    fish_categories = set()
    for cls_idx, cls_name in model_class_names.items():
        if any(keyword.lower() in cls_name.lower() for keyword in fish_keywords):
            fish_categories.add(cls_name)
            fish_categories.add(t(cls_name))
    fish_categories.update({"健康", "亚健康", "患病", "health", "Subhealthy", "Diseased"})
    in_path = Path("traj_input_tmp.mp4")
    in_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        return {"success": False, "message": t('cannot_read_video_file'), "total_distance": 0, "average_speed": 0, "video_duration": 0, "total_frames": 0, "processed_video_path": ""}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_video_path = Path(f"traj_processed_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(processed_video_path), fourcc, fps, (w, h))
    progress_bar = st.progress(0)
    status_text = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        if max_frames and total_frames > max_frames:
            break
        progress = min(total_frames / total_frames_total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{t('tracking_processing')} {total_frames}/{total_frames_total}")
        try:
            r = MODELS[model_key].predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
        except Exception as e:
            cap.release()
            out.release()
            return {"success": False, "message": f"{t('frame_inference_failed')} {str(e)}", "total_distance": 0, "average_speed": 0, "video_duration": 0, "total_frames": total_frames, "processed_video_path": ""}
        frame_with_detect = r.plot()
        current_center = None
        max_conf = 0.0
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                cls_name = r.names.get(cls_idx, "")
                if cls_name in fish_categories:
                    conf_score = float(box.conf.item())
                    if conf_score > max_conf:
                        max_conf = conf_score
                        xyxy = box.xyxy.cpu().numpy()[0]
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        current_center = (center_x, center_y)
        if current_center is not None:
            if prev_center is not None:
                distance = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])
                total_distance += distance
                cv2.line(frame_with_detect, prev_center, current_center, (0, 0, 255), 2)
            trajectory_points.append(current_center)
            cv2.circle(frame_with_detect, current_center, 5, (255, 0, 0), -1)
            prev_center = current_center
        out.write(frame_with_detect)
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    video_duration = total_frames / fps if fps > 0 else 0
    average_speed = total_distance / video_duration if video_duration > 0 else 0
    if total_distance == 0:
        return {"success": True, "message": f"{t('no_fish_detected')}", "total_distance": 0, "average_speed": 0, "video_duration": round(video_duration, 2), "total_frames": total_frames, "processed_video_path": str(processed_video_path) if processed_video_path.exists() else ""}
    return {"success": True, "message": "轨迹分析完成", "total_distance": round(total_distance, 2), "average_speed": round(average_speed, 2), "video_duration": round(video_duration, 2), "total_frames": total_frames, "processed_video_path": str(processed_video_path) if processed_video_path.exists() else ""}

def save_table_to_excel(df: pd.DataFrame, filename: str) -> Path:
    out = Path(filename).with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="detections", index=False)
    return out

def zip_files(files: list[Path], out_zip: Path) -> Path:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                zf.write(f, arcname=f.name)
    return out_zip

@st.cache_resource
def build_fuzzy_sim():
    behavior = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'behavior')
    surf = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'surf')
    patho = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'patho')
    risk = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'risk')
    behavior['healthy'] = fuzz.trimf(behavior.universe, [1, 1, 1.5])
    behavior['subhealthy'] = fuzz.trimf(behavior.universe, [1.5, 2, 2.5])
    behavior['diseased'] = fuzz.trimf(behavior.universe, [2.5, 3, 4])
    surf['healthy'] = fuzz.trimf(surf.universe, [1, 1, 2])
    surf['diseased'] = fuzz.trimf(surf.universe, [2, 3, 4])
    patho['absent'] = fuzz.trimf(patho.universe, [1, 1, 2])
    patho['present'] = fuzz.trimf(patho.universe, [2, 3, 4])
    risk['health'] = fuzz.trimf(risk.universe, [0, 1, 1.5])
    risk['subhealth'] = fuzz.trimf(risk.universe, [1.5, 2, 2.5])
    risk['diseased'] = fuzz.trimf(risk.universe, [2.5, 3, 4])
    risk.defuzzify_method = 'centroid'
    rules = [ctrl.Rule(behavior['healthy'] & surf['healthy'] & patho['absent'], risk['health']), ctrl.Rule(behavior['healthy'] & surf['healthy'] & patho['present'], risk['subhealth']), ctrl.Rule(behavior['diseased'], risk['diseased']), ctrl.Rule(surf['diseased'] & patho['present'], risk['diseased']), ctrl.Rule(behavior['subhealthy'], risk['subhealth']), ctrl.Rule(surf['diseased'] & patho['absent'], risk['subhealth'])]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def fuzzy_predict(behavior_val: float, surf_val: float, patho_val: float) -> dict:
    try:
        sim = build_fuzzy_sim()
        sim.input['behavior'] = behavior_val
        sim.input['surf'] = surf_val
        sim.input['patho'] = patho_val
        sim.compute()
        v = float(sim.output['risk'])
        if st.session_state.language == 'zh':
            status = "健康" if v < 1.5 else ("亚健康" if v < 2.5 else "患病")
        else:
            status = t("healthy") if v < 1.5 else (t("subhealthy") if v < 2.5 else t("diseased"))
        return {"risk_value": round(v, 1), "risk_status": status}
    except Exception as e:
        st.warning(f"{t('fuzzy_calc_error')}{str(e)}")
        return {"risk_value": 2.0, "risk_status": t("subhealthy")}

# ====================== 样式 ======================
st.markdown("""<style>.app-header {background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); color:white; border-radius:16px; padding:16px; text-align:center;}.app-title {font-size:30px; font-weight:bold;}.traj-card {background:#f0f8ff; border:1px solid #b8d4ff; border-radius:12px; padding:12px; margin:8px 0;}.traj-metric {font-size:18px; font-weight:bold; color:#2563eb;}.healthy {color:#48bb78;}.subhealthy {color:#ed8936;}.diseased {color:#e53e3e;}</style>""", unsafe_allow_html=True)

# ====================== 界面 ======================
with st.sidebar:
    st.markdown(f"### 🎓 {t('sidebar_university')}")
    st.divider()
    st.header(t('sidebar_model'))
    model_options = {"Ich": t("Ich"), "Tomont": t("Tomont"), "Behavior": t("Behavior"), "CiSurface": t("CiSurface"), "CiTomont": t("CiTomont"), "CroakerBehavior": t("CroakerBehavior")}
    available_models = {k: model_options[k] for k in MODELS.keys()}
    default_model = "Ich" if "Ich" in available_models else list(available_models.keys())[0]
    model_value = st.selectbox(t('sidebar_model_type'), options=list(available_models.keys()), format_func=lambda x: available_models[x], index=list(available_models.keys()).index(default_model))
    st.markdown(f"✅ 当前模型：**{available_models[model_value]}**")

tab_img, tab_folder, tab_video, tab_camera, tab_tracking, tab_fuzzy = st.tabs([t('tab_image'), t('tab_batch'), t('tab_video'), t('tab_camera'), t('tab_tracking'), t('tab_fuzzy')])

with tab_img:
    st.markdown(f"#### {t('tab_image')}")
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.file_uploader(t('image_upload'), type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_upload_single")
        if img_file:
            st.image(Image.open(img_file), use_column_width=True)
    with col2:
        run = st.button(t('image_run'), type="primary", disabled=img_file is None or model_value is None, key="btn_img_single")
        if run and img_file:
            with st.spinner("检测中..."):
                det_img, df = predict_on_image(img_file.getvalue(), model_value)
            st.image(det_img, use_column_width=True)
            if not df.empty:
                st.dataframe(df)

with tab_folder:
    st.markdown("#### 批量检测")
    files = st.file_uploader("上传多张图片", accept_multiple_files=True, key="batch_upload_imgs")
    go = st.button("开始批量检测", disabled=not files or model_value is None, key="btn_batch_run")
    if go and files:
        all_tables = []
        out_imgs = []
        progress = st.progress(0)
        for i, f in enumerate(files):
            det_img, df = predict_on_image(f.getvalue(), model_value)
            if not df.empty:
                df[t("path")] = f.name
                all_tables.append(df)
            out_path = Path(f"{f.name}_detect.jpg")
            det_img.save(out_path)
            out_imgs.append(out_path)
            progress.progress((i+1)/len(files))
        df_all = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
        st.dataframe(df_all)

with tab_video:
    st.markdown("#### 视频检测")
    vid_file = st.file_uploader("上传检测视频", type=["mp4", "mov", "avi", "mkv"], key="video_upload_detect")
    run = st.button("开始检测", disabled=vid_file is None or not CV2_OK or model_value is None, key="btn_video_detect")
    if run and vid_file:
        with st.spinner("处理中..."):
            out_path = process_video(vid_file.getvalue(), model_value)
        st.success("处理完成")
        st.download_button("下载视频", open(out_path, "rb").read(), file_name=out_path.name, key="dl_video_result")

with tab_camera:
    st.markdown("#### 摄像头检测")
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False
    if not st.session_state.cam_on:
        if st.button("打开摄像头", key="btn_cam_open"):
            st.session_state.cam_on = True
            st.rerun()
    else:
        if st.button("关闭摄像头", key="btn_cam_close"):
            st.session_state.cam_on = False
            st.rerun()
        snap = st.camera_input("拍照", key="cam_input_take")
        if snap and st.button("检测", key="btn_cam_detect"):
            det_img, df = predict_on_image(snap.getvalue(), model_value)
            st.image(det_img)
            st.dataframe(df)

with tab_tracking:
    st.markdown("#### 🐠 轨迹分析")
    vid_file = st.file_uploader("上传轨迹分析视频", type=["mp4", "mov", "avi", "mkv"], key="video_upload_traj")
    conf = st.slider("置信度", 0.1, 1.0, 0.3, key="slider_traj_conf")
    time_period = st.selectbox("时间段", [t("日间"), t("夜间")], key="select_traj_time")
    run = st.button("开始分析", disabled=vid_file is None or not CV2_OK or model_value is None, key="btn_traj_run")
    if run and vid_file:
        res = calculate_fish_trajectory(vid_file.getvalue(), model_value, conf)
        if res["success"]:
            health = get_health_status(res["average_speed"], time_period)
            st.metric("总路程", res["total_distance"])
            st.metric("平均速度", res["average_speed"])
            st.metric("健康状态", health)

with tab_fuzzy:
    st.markdown("#### 模糊预测")
    col1, col2, col3 = st.columns(3)
    with col1:
        b = st.selectbox("行为", ["健康", "亚健康", "患病"], key="fuzzy_behavior")
    with col2:
        s = st.selectbox("体表", ["健康", "患病"], key="fuzzy_surface")
    with col3:
        p = st.selectbox("病原", ["不存在", "存在"], key="fuzzy_pathogen")
    bv = 1 if b == "健康" else 2 if b == "亚健康" else 3
    sv = 1 if s == "健康" else 3
    pv = 1 if p == "不存在" else 3
    if st.button("预测", key="btn_fuzzy_predict"):
        r = fuzzy_predict(bv, sv, pv)
        st.success(f"风险值：{r['risk_value']}，状态：{r['risk_status']}")
