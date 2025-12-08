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
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import os

# ====================== 基础配置 ======================
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

# 翻译字典
translations = {
    'zh': {
        'tab_tracking': '📍 轨迹跟踪',
        'tracking_title': '金鱼运动轨迹分析',
        'tracking_upload': '上传视频文件',
        'tracking_run': '🚀 开始轨迹分析',
        'tracking_processing': '正在分析视频轨迹...',
        'total_distance': '总路程（像素）',
        'average_speed': '平均运动速度（像素/秒）',
        'video_duration': '视频时长（秒）',
        'total_frames': '总帧数',
        'no_fish_detected': '未检测到金鱼，无法计算轨迹数据',
        '行为': '行为分析模型',
        'page_title': 'YOLO病害检测',
        'header_title': '鱼类寄生虫病检测',
        'header_subtitle': '图片 / 批量 / 视频 / 摄像头 / 轨迹跟踪 / 模糊预测 — 一站式检测台',
        'sidebar_university': '宁波大学 · 病害实验室',
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
        'video_upload': '上传视频',
        'video_run': '🚀 开始视频检测',
        'video_disabled': '当前环境未加载 OpenCV，视频功能禁用',
        'video_processing': '视频处理中...',
        'video_download': '下载处理后视频',
        'camera_title': '📷 摄像头检测',
        'camera_caption': '点击打开/关闭摄像头',
        'camera_open': '🎬 打开摄像头',
        'camera_close': '⏹ 关闭摄像头',
        'camera_not_started': '摄像头未开启',
        'camera_shot': '拍照',
        'camera_detect': '检测',
        'fuzzy_title': '🧮 模糊预测',
        'fuzzy_input': '输入指标参数',
        'fuzzy_day': '日间行为（1~3）',
        'fuzzy_night': '夜间行为（1~3）',
        'fuzzy_surface': '体表特征（1~3）',
        'fuzzy_pathogen': '病原特征（1~3）',
        'fuzzy_predict': '🧪 预测',
        'fuzzy_result': '风险值: {risk_value}，状态: {risk_status}',
        'Ich': '多子小瓜虫病',
        'Tomont': '包囊',
        'healthy': '健康',
        'subhealthy': '亚健康',
        'diseased': '患病',
        'category': '类别',
        'confidence': '置信度',
        'location': '位置',
        'path': '路径',
        'model_not_found': '模型文件不存在：{p}（{k}模型）',
        'model_loaded': '成功加载模型：{k} -> {p}'
    },
    'en': {
        'tab_tracking': '📍 Trajectory Tracking',
        'tracking_title': 'Goldfish Motion Trajectory Analysis',
        'tracking_upload': 'Upload Video File',
        'tracking_run': '🚀 Start Trajectory Analysis',
        'tracking_processing': 'Analyzing trajectory...',
        'total_distance': 'Total Distance (pixels)',
        'average_speed': 'Average Speed (pixels/sec)',
        'video_duration': 'Video Duration (sec)',
        'total_frames': 'Total Frames',
        'no_fish_detected': 'No goldfish detected',
        '行为': 'Behavior Analysis Model',
        'page_title': 'YOLO Disease Detection',
        'header_title': 'Fish Parasitic Disease Detection',
        'header_subtitle': 'Image/Batch/Video/Camera/Trajectory/Fuzzy Prediction',
        'sidebar_university': 'Ningbo University · Lab',
        'sidebar_model': '🧠 Model & Params',
        'sidebar_model_type': 'Model Type',
        'sidebar_current_model': 'Current Model:',
        'tab_image': '🖼️ Image Detection',
        'tab_batch': '🗂️ Batch Detection',
        'tab_video': '🎞️ Video Detection',
        'tab_camera': '📷 Camera Detection',
        'tab_fuzzy': '🧮 Fuzzy Prediction',
        'image_original': 'Original Image',
        'image_detection': 'Detection Result',
        'image_upload': 'Upload Image',
        'image_run': '🚀 Start Detection',
        'image_result': 'Result',
        'image_download_excel': 'Download Excel',
        'image_download_img': 'Download Annotated Image',
        'batch_upload': 'Upload Multiple Images',
        'batch_run': '🚀 Start Batch Detection',
        'batch_processing': 'Processing:',
        'batch_total': 'Total:',
        'batch_no_results': 'No targets detected',
        'batch_download_excel': '📥 Download Excel',
        'batch_download_zip': '🗜️ Download Images ZIP',
        'video_upload': 'Upload Video',
        'video_run': '🚀 Start Video Detection',
        'video_disabled': 'OpenCV not loaded, video disabled',
        'video_processing': 'Processing video...',
        'video_download': 'Download Processed Video',
        'camera_title': '📷 Camera Detection',
        'camera_caption': 'Click to open/close camera',
        'camera_open': '🎬 Open Camera',
        'camera_close': '⏹ Close Camera',
        'camera_not_started': 'Camera not started',
        'camera_shot': 'Take Photo',
        'camera_detect': 'Detect',
        'fuzzy_title': '🧮 Fuzzy Prediction',
        'fuzzy_input': 'Input Parameters',
        'fuzzy_day': 'Day Behavior (1~3)',
        'fuzzy_night': 'Night Behavior (1~3)',
        'fuzzy_surface': 'Surface Feature (1~3)',
        'fuzzy_pathogen': 'Pathogen Feature (1~3)',
        'fuzzy_predict': '🧪 Predict',
        'fuzzy_result': 'Risk Value: {risk_value}, Status: {risk_status}',
        'Ich': 'Ichthyophthirius',
        'Tomont': 'Tomont',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'category': 'Category',
        'confidence': 'Confidence',
        'location': 'Location',
        'path': 'Path',
        'model_not_found': 'Model not found: {p} ({k} model)',
        'model_loaded': 'Loaded model: {k} -> {p}'
    }
}

def t(key):
    return translations[st.session_state.language].get(key, key)

st.set_page_config(page_title=t('page_title'), page_icon="🧪", layout="wide")

# ====================== 模型加载 ======================
BASE_DIR = Path(__file__).parent
MODEL_PATHS = {
    "Ich": BASE_DIR / "best.pt",
    "Tomont": BASE_DIR / "tomont.best.pt",
    "行为": BASE_DIR / "guijibest.pt"
}
DEFAULT_CONF = 0.6

@st.cache_resource
def load_models():
    models = {}
    for k, p in MODEL_PATHS.items():
        if p.exists():
            try:
                models[k] = YOLO(str(p))
                st.success(t('model_loaded').format(k=k, p=str(p)))
            except Exception as e:
                st.error(f"加载模型{k}失败：{str(e)}")
        else:
            st.error(t('model_not_found').format(k=k, p=str(p)))
    if not models and MODEL_PATHS["Ich"].exists():
        models["Ich"] = YOLO(str(MODEL_PATHS["Ich"]))
        st.warning("仅加载Ich模型作为兜底")
    return models

MODELS = load_models()

# ====================== 工具函数 ======================
def detections_to_df(res):
    if hasattr(res, "boxes") and hasattr(res, "names"):
        rows = []
        for box in res.boxes:
            cls = int(box.cls.item())
            rows.append({
                t("category"): res.names.get(cls, str(cls)),
                t("confidence"): float(box.conf.item()),
                t("location"): [float(x) for x in box.xyxy.cpu().numpy()[0]]
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()

def predict_on_image(img_input, model_key, conf=DEFAULT_CONF):
    if isinstance(img_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, Image.Image):
        pil_img = img_input.convert("RGB")
    elif isinstance(img_input, (str, Path)):
        pil_img = Image.open(str(img_input)).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)) if CV2_OK else Image.fromarray(img_input)
    else:
        raise TypeError(f"不支持的输入类型：{type(img_input)}")

    if model_key not in MODELS:
        model_key = next(iter(MODELS.keys())) if MODELS else None
    if not model_key:
        raise RuntimeError("无可用模型")
    
    results = MODELS[model_key](source=pil_img, conf=conf, imgsz=640, verbose=False)
    det_img = results[0].plot()
    det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB) if CV2_OK else det_img
    return Image.fromarray(det_img_rgb), detections_to_df(results[0])

def process_video(video_bytes, model_key, conf=DEFAULT_CONF, max_frames=None):
    if not CV2_OK:
        raise RuntimeError(t('video_disabled'))
    
    # 临时输入文件
    in_path = BASE_DIR / f"tmp_input_{int(time.time())}.mp4"
    in_path.write_bytes(video_bytes)
    
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        in_path.unlink()
        raise RuntimeError("无法读取视频")
    
    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出文件（绝对路径）
    out_path = BASE_DIR / f"processed_{int(time.time())}.mp4"
    
    # 关键修复：使用H.264编码（avc1），兼容Streamlit预览
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 降级兼容
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # 模型推理
        results = MODELS[model_key](source=frame, conf=conf, imgsz=640, verbose=False)
        out.write(results[0].plot())
        frame_count += 1
    
    cap.release()
    out.release()
    in_path.unlink()  # 删除临时输入文件
    return out_path

# 轨迹分析核心函数（修复视频编码+字节流）
def calculate_fish_trajectory(video_bytes, model_key, conf=DEFAULT_CONF, max_frames=None):
    if not CV2_OK:
        return {
            "success": False,
            "message": t('video_disabled'),
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": "",
            "processed_video_bytes": b""
        }
    
    # 校验模型
    if model_key not in MODELS:
        model_key = next(iter(MODELS.keys())) if MODELS else None
    if not model_key:
        return {
            "success": False,
            "message": "无可用模型",
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": "",
            "processed_video_bytes": b""
        }
    
    # 临时输入文件（绝对路径）
    in_path = BASE_DIR / f"traj_input_{int(time.time())}.mp4"
    in_path.write_bytes(video_bytes)
    
    # 打开视频
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        in_path.unlink()
        return {
            "success": False,
            "message": "无法读取视频文件",
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": "",
            "processed_video_bytes": b""
        }
    
    # 视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 输出视频路径（绝对路径）
    out_path = BASE_DIR / f"traj_output_{int(time.time())}.mp4"
    
    # 关键修复：强制H.264编码
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264（Streamlit预览兼容）
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 降级
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    # 轨迹计算变量
    prev_center = None
    total_distance = 0.0
    frame_count = 0
    trajectory_points = []
    
    # 目标类别过滤
    current_model = MODELS[model_key]
    model_class_names = current_model.names
    fish_keywords = ["healthy", "subhealthy", "diseased", "健康", "亚健康", "患病", "金鱼", "fish"]
    fish_categories = set()
    for cls_idx, cls_name in model_class_names.items():
        if any(keyword.lower() in cls_name.lower() for keyword in fish_keywords):
            fish_categories.add(cls_name)
            fish_categories.add(t(cls_name))
    fish_categories.update({"健康", "亚健康", "患病", "health", "Subhealthy", "Diseased"})
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # 更新进度
        progress = min(frame_count / total_frames_total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{t('tracking_processing')} {frame_count}/{total_frames_total}")
        
        # 模型推理
        results = current_model(source=frame, conf=conf, imgsz=640, verbose=False)
        result_frame = results[0].plot()
        
        # 提取目标中心坐标
        current_center = None
        max_conf = 0.0
        if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_idx = int(box.cls.item())
                cls_name = results[0].names.get(cls_idx, "")
                if cls_name in fish_categories:
                    conf_score = float(box.conf.item())
                    if conf_score > max_conf:
                        max_conf = conf_score
                        xyxy = box.xyxy.cpu().numpy()[0]
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        current_center = (center_x, center_y)
        
        # 计算轨迹距离
        if current_center is not None:
            trajectory_points.append(current_center)
            if prev_center is not None:
                # 欧氏距离
                distance = math.hypot(
                    current_center[0] - prev_center[0],
                    current_center[1] - prev_center[1]
                )
                total_distance += distance
                # 绘制轨迹线
                cv2.line(result_frame, prev_center, current_center, (0, 0, 255), 2)
            # 绘制中心点
            cv2.circle(result_frame, current_center, 5, (255, 0, 0), -1)
            prev_center = current_center
        
        # 写入视频
        out.write(result_frame)
        frame_count += 1
    
    # 清理资源
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    # 计算统计值
    video_duration = frame_count / fps if fps > 0 else 0
    average_speed = total_distance / video_duration if video_duration > 0 else 0
    
    # 关键修复：读取视频为字节流（用于预览）
    processed_video_bytes = b""
    if out_path.exists():
        with open(out_path, "rb") as f:
            processed_video_bytes = f.read()
    
    # 调试信息
    st.info(f"""
    视频编码：{"H.264 (avc1)" if fourcc == cv2.VideoWriter_fourcc(*'avc1') else "MP4V"}
    视频路径：{str(out_path)}
    视频大小：{len(processed_video_bytes) / 1024 / 1024:.2f} MB
    目标过滤类别：{fish_categories}
    """)
    
    # 返回结果（包含字节流）
    return {
        "success": True,
        "message": f"轨迹分析完成（使用模型：{model_key}）",
        "total_distance": round(total_distance, 2),
        "average_speed": round(average_speed, 2),
        "video_duration": round(video_duration, 2),
        "total_frames": frame_count,
        "processed_video_path": str(out_path),
        "processed_video_bytes": processed_video_bytes  # 新增：视频字节流
    }

def save_table_to_excel(df, filename):
    out = Path(filename).with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="detections", index=False)
    return out

def zip_files(files, out_zip):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                zf.write(f, arcname=f.name)
    return out_zip

# 模糊预测
@st.cache_resource
def build_fuzzy_sim():
    day = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'day')
    night = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'night')
    surf = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'surf')
    patho = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'patho')
    risk = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'risk')

    for b in [day, night]:
        b['healthy'] = fuzz.trimf(b.universe, [1, 1, 1.5])
        b['subhealthy'] = fuzz.trimf(b.universe, [1.5, 2, 2.5])
        b['diseased'] = fuzz.trimf(b.universe, [2.5, 3, 4])

    surf['healthy'] = fuzz.trimf(surf.universe, [1, 1, 2])
    surf['diseased'] = fuzz.trimf(surf.universe, [2, 3, 4])
    patho['absent'] = fuzz.trimf(patho.universe, [1, 1, 2])
    patho['present'] = fuzz.trimf(patho.universe, [2, 3, 4])

    risk['health'] = fuzz.trimf(risk.universe, [0, 1, 1.5])
    risk['subhealth'] = fuzz.trimf(risk.universe, [1.5, 2, 2.5])
    risk['diseased'] = fuzz.trimf(risk.universe, [2.5, 3, 4])
    risk.defuzzify_method = 'centroid'

    rules = [
        ctrl.Rule(day['subhealthy'] & night['diseased'] & surf['healthy'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['healthy'] & night['healthy'] & surf['healthy'] & patho['absent'], risk['health']),
        ctrl.Rule(day['diseased'] | night['diseased'], risk['diseased']),
        ctrl.Rule(day['subhealthy'] | night['subhealthy'], risk['subhealth']),
        ctrl.Rule(surf['diseased'] & patho['present'], risk['diseased']),
        ctrl.Rule(surf['healthy'] & patho['absent'], risk['health']),
        ctrl.Rule(day['healthy'] & night['subhealthy'] & surf['healthy'] & patho['present'], risk['subhealth']),
        ctrl.Rule(day['subhealthy'] & night['healthy'] & surf['healthy'] & patho['present'], risk['subhealth']),
        ctrl.Rule(day['healthy'] & night['healthy'] & surf['diseased'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['healthy'] & night['healthy'] & surf['healthy'] & patho['present'], risk['subhealth']),
        ctrl.Rule(day['subhealthy'] & night['subhealthy'] & surf['healthy'] & patho['absent'], risk['health']),
        ctrl.Rule(day['subhealthy'] & night['subhealthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        ctrl.Rule(day['subhealthy'] & night['diseased'] & surf['diseased'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['diseased'] & night['subhealthy'] & surf['diseased'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['subhealthy'] & night['subhealthy'] & surf['diseased'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['healthy'] & night['subhealthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        ctrl.Rule(day['subhealthy'] & night['healthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        ctrl.Rule(day['subhealthy'] & night['subhealthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        ctrl.Rule(day['healthy'] & night['healthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        ctrl.Rule(day['diseased'] & night['diseased'] & surf['healthy'] & patho['absent'], risk['diseased']),
        ctrl.Rule(day['diseased'] & night['diseased'] & surf['diseased'] & patho['absent'], risk['diseased']),
        ctrl.Rule(day['diseased'] & night['diseased'] & surf['healthy'] & patho['present'], risk['diseased']),
        ctrl.Rule(day['diseased'] & night['diseased'] & surf['diseased'] & patho['present'], risk['diseased']),
    ]
    for r in rules:
        r.weight = 1.0
    rules[4].weight = 2
    rules[5].weight = 2

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def fuzzy_predict(day_val, night_val, surf_val, patho_val):
    sim = build_fuzzy_sim()
    sim.input['day'] = day_val
    sim.input['night'] = night_val
    sim.input['surf'] = surf_val
    sim.input['patho'] = patho_val
    sim.compute()
    v = float(sim.output['risk'])
    if st.session_state.language == 'zh':
        status = "健康" if v < 1.5 else ("亚健康" if v < 2.5 else "患病")
    else:
        status = t("healthy") if v < 1.5 else (t("subhealthy") if v < 2.5 else t("diseased"))
    return {"risk_value": round(v, 1), "risk_status": status}

# ====================== 页面样式 ======================
st.markdown("""
<style>
.app-header {
  background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
  color: white; border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
}
.card {
  background: #F6F7FB; border: 1px solid #E5E7EB; border-radius: 14px; padding: 14px; margin-bottom: 12px;
}
.traj-card {
  background: #f0f8ff; border: 1px solid #b8d4ff; border-radius: 12px; padding: 16px; margin: 8px 0;
}
.traj-metric {
  font-size: 18px; font-weight: 600; color: #2563eb;
}
.stButton>button { border-radius: 10px; }
.block-container { padding-top: 0.6rem; padding-bottom: 1rem; }
.badge {
  background: #EEF2FF; color:#3730A3; border:1px solid #E0E7FF;
  padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# 隐藏默认组件
st.markdown("""
<style>
header[data-testid="stHeader"], #MainMenu, footer {visibility: hidden;}
[data-testid="stAppViewContainer"] .main .block-container { padding-top: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ====================== 侧边栏 ======================
with st.sidebar:
    # 语言切换
    col1, col2 = st.columns(2)
    with col1:
        if st.button('中文', use_container_width=True):
            st.session_state.language = 'zh'
            st.rerun()
    with col2:
        if st.button('English', use_container_width=True):
            st.session_state.language = 'en'
            st.rerun()
    
    st.markdown(f"### 🎓 {t('sidebar_university')}")
    st.divider()
    
    # 全局模型选择
    st.header(t('sidebar_model'))
    if MODELS:
        model_options = {k: t(k) for k in MODELS.keys()}
        model_value = st.selectbox(
            t('sidebar_model_type'),
            options=list(model_options.keys()),
            format_func=lambda x: f"{x}（{model_options[x]}）",
            index=0 if "Ich" in model_options else 0
        )
        st.markdown(f"<span class='badge'>{t('sidebar_current_model')} <b>{model_value}</b></span>", unsafe_allow_html=True)
    else:
        model_value = None
        st.error("无可用模型，请检查模型文件！")

# ====================== 顶部标题 ======================
st.markdown(f"""
<div class="app-header">
  <h1>{t('header_title')}</h1>
  <p>{t('header_subtitle')}</p>
</div>
""", unsafe_allow_html=True)

# ====================== 标签页 ======================
tab_img, tab_folder, tab_video, tab_camera, tab_tracking, tab_fuzzy = st.tabs([
    t('tab_image'), t('tab_batch'), t('tab_video'), t('tab_camera'), t('tab_tracking'), t('tab_fuzzy')
])

# -------------------------- 图片检测 --------------------------
with tab_img:
    st.markdown(f"#### {t('tab_image')}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<div class='card'><b>{t('image_original')}</b></div>", unsafe_allow_html=True)
        img_file = st.file_uploader(t('image_upload'), type=["jpg","jpeg","png","bmp","webp"], key="single_img")
        if img_file:
            st.image(Image.open(img_file), caption=t('image_original'), use_column_width=True)
    
    with col2:
        st.markdown(f"<div class='card'><b>{t('image_detection')}</b></div>", unsafe_allow_html=True)
        run_btn = st.button(t('image_run'), type="primary", use_container_width=True, disabled=not (img_file and model_value))
        
        if run_btn and img_file and model_value:
            with st.spinner(t('batch_processing')):
                det_img, df = predict_on_image(img_file.getvalue(), model_value)
            
            st.image(det_img, caption=t('image_result'), use_column_width=True)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(t('image_download_excel'), use_container_width=True):
                        xlsx_path = save_table_to_excel(df, "image_result.xlsx")
                        st.download_button(
                            t('image_download_excel'),
                            data=open(xlsx_path, "rb").read(),
                            file_name=xlsx_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                with col_b:
                    if st.button(t('image_download_img'), use_container_width=True):
                        bio = io.BytesIO()
                        det_img.save(bio, format="JPEG")
                        st.download_button(
                            t('image_download_img'),
                            data=bio.getvalue(),
                            file_name="image_result.jpg",
                            mime="image/jpeg"
                        )

# -------------------------- 批量检测 --------------------------
with tab_folder:
    st.markdown(f"#### {t('tab_batch')}")
    files = st.file_uploader(
        t('batch_upload'),
        type=["jpg","jpeg","png","bmp","webp"],
        accept_multiple_files=True,
        key="batch_imgs"
    )
    
    run_btn = st.button(t('batch_run'), type="primary", use_container_width=True, disabled=not (files and model_value))
    
    if run_btn and files and model_value:
        all_dfs = []
        out_imgs = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(files):
            status.text(f"{t('batch_processing')} {f.name} ({i+1}/{len(files)})")
            with st.spinner(f"处理 {f.name}..."):
                det_img, df = predict_on_image(f.getvalue(), model_value)
                if not df.empty:
                    df[t('path')] = f.name
                    all_dfs.append(df)
                
                # 保存检测后图片
                out_path = BASE_DIR / f"{Path(f.name).stem}_detect.jpg"
                det_img.save(out_path)
                out_imgs.append(out_path)
            
            progress.progress((i+1)/len(files))
        
        # 结果展示
        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            st.dataframe(df_all, use_container_width=True)
            
            # Excel下载
            xlsx_path = save_table_to_excel(df_all, "batch_result.xlsx")
            st.download_button(
                t('batch_download_excel'),
                data=open(xlsx_path, "rb").read(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # 图片打包下载
        if out_imgs:
            zip_path = zip_files(out_imgs, BASE_DIR / "batch_images.zip")
            st.download_button(
                t('batch_download_zip'),
                data=open(zip_path, "rb").read(),
                file_name=zip_path.name,
                mime="application/zip",
                use_container_width=True
            )
        
        progress.empty()
        status.empty()

# -------------------------- 视频检测 --------------------------
with tab_video:
    st.markdown(f"#### {t('tab_video')}")
    vid_file = st.file_uploader(t('video_upload'), type=["mp4","mov","avi","mkv"], key="single_video")
    
    run_btn = st.button(t('video_run'), type="primary", use_container_width=True, disabled=not (vid_file and model_value and CV2_OK))
    
    if not CV2_OK:
        st.warning(t('video_disabled'))
    
    if run_btn and vid_file and model_value and CV2_OK:
        with st.spinner(t('video_processing')):
            out_path = process_video(vid_file.getvalue(), model_value)
        
        st.video(str(out_path))
        st.download_button(
            t('video_download'),
            data=open(out_path, "rb").read(),
            file_name=out_path.name,
            mime="video/mp4",
            use_container_width=True
        )

# -------------------------- 摄像头检测 --------------------------
with tab_camera:
    st.markdown(f"#### {t('camera_title')}")
    st.caption(t('camera_caption'))
    
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(t('camera_open'), type="primary", disabled=st.session_state.cam_on):
            st.session_state.cam_on = True
            st.rerun()
    with col_b:
        if st.button(t('camera_close'), type="secondary", disabled=not st.session_state.cam_on):
            st.session_state.cam_on = False
            st.rerun()
    
    if st.session_state.cam_on:
        snap = st.camera_input(t('camera_shot'), key="cam_snap")
        if snap and model_value:
            if st.button(t('camera_detect'), type="primary", use_container_width=True):
                with st.spinner(t('batch_processing')):
                    det_img, df = predict_on_image(snap.getvalue(), model_value)
                
                st.image(det_img, caption=t('image_result'), use_column_width=True)
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
    else:
        st.info(t('camera_not_started'))

# -------------------------- 轨迹跟踪（核心修复） --------------------------
with tab_tracking:
    st.markdown(f"#### {t('tracking_title')}")
    
    # 轨迹模型选择（独立于全局模型）
    if MODELS:
        model_tracking_options = {k: t(k) for k in MODELS.keys()}
        model_tracking = st.selectbox(
            "选择轨迹分析模型",
            options=list(model_tracking_options.keys()),
            format_func=lambda x: f"{x}（{model_tracking_options[x]}）",
            index=0 if "Ich" in model_tracking_options else 0,
            key="tracking_model"
        )
        st.markdown(f"<div class='card'>当前使用模型：{model_tracking}（{model_tracking_options[model_tracking]}）</div>", unsafe_allow_html=True)
    else:
        model_tracking = None
        st.error("无可用模型，请先检查模型文件！")
    
    # 视频上传
    vid_file = st.file_uploader(
        t('tracking_upload'),
        type=["mp4","mov","avi","mkv"],
        key="tracking_video",
        help="支持MP4/MOV/AVI/MKV，建议时长≤1分钟"
    )
    
    # 视频预览分栏
    if vid_file:
        st.markdown("### 🎬 视频预览")
        col_origin, col_traj = st.columns(2)
        
        # 原始视频
        with col_origin:
            st.markdown("#### 原始视频")
            st.video(vid_file)
        
        # 轨迹视频预览区（初始提示）
        with col_traj:
            st.markdown("#### 带轨迹跟踪的视频")
            traj_video_placeholder = st.empty()
            traj_video_placeholder.info("点击“开始轨迹分析”后，此处将显示带轨迹的视频预览")
    else:
        traj_video_placeholder = None
        st.info("请先上传视频文件，上传后将显示原始视频和轨迹视频预览区域")
    
    # 参数设置
    st.markdown("### ⚙️ 分析参数")
    col_conf, col_frames = st.columns(2)
    with col_conf:
        conf_threshold = st.slider(
            "检测置信度阈值",
            min_value=0.05, max_value=1.0, value=0.3, step=0.05,
            help="阈值越低，检测到的目标越多（可能包含误检）"
        )
    with col_frames:
        max_frames = st.number_input(
            "最大分析帧数（0=无限制）",
            min_value=0, max_value=10000, value=0, step=100
        )
    
    # 开始分析按钮
    run_tracking = st.button(
        t('tracking_run'),
        type="primary",
        disabled=(not vid_file or not CV2_OK or not model_tracking),
        use_container_width=True
    )
    
    if not CV2_OK:
        st.warning(t('video_disabled'))
    
    # 执行轨迹分析
    if run_tracking and vid_file and CV2_OK and model_tracking:
        # 调用修复后的轨迹分析函数
        result = calculate_fish_trajectory(
            video_bytes=vid_file.getvalue(),
            model_key=model_tracking,
            conf=conf_threshold,
            max_frames=max_frames if max_frames > 0 else None
        )
        
        # 结果展示
        st.markdown("### 📊 轨迹分析结果")
        if result["success"]:
            # 轨迹数据卡片
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="traj-card">
                    <p>总路程（像素）</p>
                    <p class="traj-metric">{result['total_distance']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="traj-card">
                    <p>平均速度（像素/秒）</p>
                    <p class="traj-metric">{result['average_speed']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="traj-card">
                    <p>视频时长（秒）</p>
                    <p class="traj-metric">{result['video_duration']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="traj-card">
                    <p>分析帧数</p>
                    <p class="traj-metric">{result['total_frames']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 关键修复：使用字节流预览视频
            if result["processed_video_bytes"] and traj_video_placeholder:
                traj_video_placeholder.empty()
                # 直接传入字节流（避免路径问题）
                traj_video_placeholder.video(result["processed_video_bytes"])
            
            # 下载按钮
            if result["processed_video_path"] and Path(result["processed_video_path"]).exists():
                st.markdown("### 📥 视频下载")
                st.download_button(
                    label="下载带轨迹的检测视频",
                    data=result["processed_video_bytes"],  # 使用字节流下载
                    file_name=f"traj_video_{model_tracking}_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            # 提示信息
            if result["total_distance"] == 0:
                st.info(result["message"])
            else:
                st.success(result["message"])
        else:
            # 失败处理
            if traj_video_placeholder:
                traj_video_placeholder.empty()
                traj_video_placeholder.error(f"轨迹分析失败：{result['message']}")
            st.error(f"分析失败：{result['message']}")

# -------------------------- 模糊预测 --------------------------
with tab_fuzzy:
    st.markdown(f"#### {t('fuzzy_title')}")
    st.markdown(f"<div class='card'><b>{t('fuzzy_input')}</b></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        day_val = st.number_input(t('fuzzy_day'), min_value=1.0, max_value=3.0, value=3.0, step=1.0)
        night_val = st.number_input(t('fuzzy_night'), min_value=1.0, max_value=3.0, value=1.0, step=1.0)
    with col2:
        surf_val = st.number_input(t('fuzzy_surface'), min_value=1.0, max_value=3.0, value=3.0, step=1.0)
        patho_val = st.number_input(t('fuzzy_pathogen'), min_value=1.0, max_value=3.0, value=3.0, step=1.0)
    
    if st.button(t('fuzzy_predict'), type="primary", use_container_width=True):
        result = fuzzy_predict(day_val, night_val, surf_val, patho_val)
        st.success(t('fuzzy_result').format(risk_value=result['risk_value'], risk_status=result['risk_status']))
