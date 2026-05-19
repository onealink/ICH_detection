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
import math  # 新增：用于计算欧氏距离


try:
    import cv2  # noqa: F401

    CV2_OK = True
except Exception:
    CV2_OK = False
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

# ====================== 语言配置（完整中英文翻译字典） ======================
if 'language' not in st.session_state:
    st.session_state.language = 'zh'  # 默认中文

# 翻译字典（关键修改：移除日间/夜间，新增行为特征）
translations = {
    'zh': {
        # 核心轨迹分析翻译
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
        # 模型相关翻译（关键修改：Behavior对应中文）
        'model_loaded': '成功加载模型：{k} -> {p}',
        'model_not_found': '模型文件不存在：{p}（{k}模型）',
        'model_switch': '模型{k}不存在，已切换为{default_model}',
        'no_available_model': '无可用模型，请检查模型文件路径！',
        'fallback_model': '所有模型加载失败，已兜底加载Ich模型',
        # 轨迹分析控件翻译
        'conf_threshold': '检测置信度阈值（降低以检测更多目标）',
        'conf_threshold_help': '阈值越低，检测到的目标越多（可能包含误检）',
        'max_frames': '最大分析帧数（0=无限制）',
        'max_frames_help': '设置最大分析帧数，0表示分析全部帧',
        'original_video': '原始视频',
        'analysis_results': '分析结果',
        'processed_video_download': '处理后视频下载',
        'download_traj_video': '下载带轨迹的检测视频',
        'video_format_help': '支持常见视频格式，建议时长不超过1分钟以保证分析速度',
        # 错误提示翻译
        'cannot_read_video': '无法读取视频',
        'cannot_read_video_file': '无法读取视频文件',
        'frame_inference_failed': '帧推理失败:',
        'analysis_failed': '分析失败：',
        'video_process_complete': '视频处理完成！',
        # 建议文本翻译
        'suggestions': '建议：1.降低置信度阈值 2.确认视频中有鱼类 3.检查模型类别是否匹配',
        'model_label': '模型：',
        'health_status_label': '健康程度：',
        # 模型名称翻译（关键修改：Behavior对应中文）
        'Ich': '多子小瓜虫病体表病征',
        'Tomont': '多子小瓜虫包囊',
        'Behavior': '金鱼游动行为分析',
        'CiSurface': '刺激隐核虫病体表病症',
        'CiTomont': '刺激隐核虫包囊',
        'CroakerBehavior': '大黄鱼游动行为分析',
        # 模糊预测翻译（核心修改：合并为行为特征）
        'fuzzy_behavior': '行为特征',  # 新增：合并日间/夜间为行为特征
        'fuzzy_surface': '体表特征',
        'fuzzy_pathogen': '病原存在性',
        'healthy': '健康',
        'subhealthy': '亚健康',
        'diseased': '患病',
        'pathogen_absent': '不存在',
        'pathogen_present': '存在',
        # 原有基础翻译
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
        'video_upload': '上传视频',
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
        # 新增：模糊预测异常提示
        'fuzzy_calc_error': '模糊计算异常，已使用默认值：'
    },
    'en': {
        # 核心轨迹分析翻译
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
        # 模型相关翻译
        'model_loaded': 'Successfully loaded model: {k} -> {p}',
        'model_not_found': 'Model file not found: {p} ({k} model)',
        'model_switch': 'Model {k} does not exist, switched to {default_model}',
        'no_available_model': 'No available models, please check model file path!',
        'fallback_model': 'All models failed to load, fallback to Ich model',
        # 轨迹分析控件翻译
        'conf_threshold': 'Detection Confidence Threshold (lower to detect more targets)',
        'conf_threshold_help': 'Lower threshold detects more targets (may include false detections)',
        'max_frames': 'Maximum Analysis Frames (0=unlimited)',
        'max_frames_help': 'Set maximum frames to analyze, 0 means analyze all frames',
        'original_video': 'Original Video',
        'analysis_results': 'Analysis Results',
        'processed_video_download': 'Processed Video Download',
        'download_traj_video': 'Download video with trajectory detection',
        'video_format_help': 'Supports common video formats, recommended duration ≤ 1 minute for speed',
        # 错误提示翻译
        'cannot_read_video': 'Cannot read video',
        'cannot_read_video_file': 'Cannot read video file',
        'frame_inference_failed': 'Frame inference failed:',
        'analysis_failed': 'Analysis failed: ',
        'video_process_complete': 'Video processing completed!',
        # 建议文本翻译
        'suggestions': 'Suggestions: 1.Lower confidence threshold 2.Confirm video contains fish 3.Check model category matching',
        'model_label': 'Model: ',
        'health_status_label': 'Health Status: ',
        # 模型名称翻译（关键修改：Behavior对应英文）
        'Ich': 'Ichthyophthirius Surface Symptoms',
        'Tomont': 'Ichthyophthirius Tomont',
        'Behavior': 'Goldfish Swimming Behavior Analysis',
        'CiSurface': 'Cryptocaryon irritans Surface Symptoms',
        'CiTomont': 'Cryptocaryon irritans Tomont',
        'CroakerBehavior': 'Large Yellow Croaker Swimming Behavior Analysis',
        # 模糊预测翻译（核心修改：合并为行为特征）
        'fuzzy_behavior': 'Behavior Feature',  # 新增：合并日间/夜间为行为特征
        'fuzzy_surface': 'Surface Features',
        'fuzzy_pathogen': 'Pathogen Existence',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'pathogen_absent': 'Absent',
        'pathogen_present': 'Present',
        # 原有基础翻译
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
        # 新增：模糊预测异常提示
        'fuzzy_calc_error': 'Fuzzy calculation error, using default value: '
    }
}

# 获取当前语言翻译
def t(key):
    return translations[st.session_state.language].get(key, key)

# ====================== 健康程度计算函数 ======================
def get_health_status(average_speed: float, time_period: str) -> str:
    """
    根据平均速度和时间段判断健康程度
    :param average_speed: 平均运动速度（像素/秒）
    :param time_period: 时间段（日间/Daytime 或 夜间/Nighttime）
    :return: 健康程度描述
    """
    # 统一判断逻辑（兼容中英文）
    is_daytime = time_period in [t("daytime"), "日间", "Daytime"]
    
    if st.session_state.language == 'zh':
        healthy = "健康"
        subhealthy = "亚健康"
        diseased = "患病"
    else:
        healthy = "Healthy"
        subhealthy = "Subhealthy"
        diseased = "Diseased"
    
    # 日间判断规则
    if is_daytime:
        if average_speed > 15:
            return healthy
        elif 10 <= average_speed <= 15:
            return subhealthy
        else:
            return diseased
    # 夜间判断规则
    else:
        if average_speed > 10:
            return healthy
        elif 5 <= average_speed <= 10:
            return subhealthy
        else:
            return diseased

# ====================== 页面配置 ======================
st.set_page_config(page_title=t('page_title'), page_icon="🧪", layout="wide")
# ====================== 模型加载（关键修改：模型键改为Behavior） ======================
# ========== 先加这 3 行调试代码，看真实路径！ ==========
import os
st.write("### 当前运行目录:", os.getcwd())
st.write("### __file__ 真实路径:", __file__)
# ======================================================

BASE_DIR = Path(__file__).parent
st.write("### 程序认为的模型目录:", BASE_DIR)  # 再打印这个
# ====================== 模型加载（强制使用当前目录，彻底解决路径问题） ======================
# 直接使用文件名，不依赖任何路径
WEIGHTS = "best.pt"
TOMONT_WEIGHTS = "tomont.best.pt"
BEHAVIOR_WEIGHTS = "guijibest.pt"

CI_SURFACE_WEIGHTS = "cybest.pt"
CI_TOMONT_WEIGHTS = "cibest.pt"
CROAKER_BEHAVIOR_WEIGHTS = "cyguijibest.pt"

IMG_DIR = "img"

# 模型路径字典
MODEL_PATHS = {
    "Ich": WEIGHTS,
    "Tomont": TOMONT_WEIGHTS,
    "Behavior": BEHAVIOR_WEIGHTS,
    "CiSurface": CI_SURFACE_WEIGHTS,
    "CiTomont": CI_TOMONT_WEIGHTS,
    "CroakerBehavior": CROAKER_BEHAVIOR_WEIGHTS,
}
DEFAULT_CONF = 0.6

@st.cache_resource
def load_models():
    models = {}
    for k, p in MODEL_PATHS.items():
        if not Path(p).exists():
            st.error(t('model_not_found').format(p=p, k=k))
        else:
            try:
                models[k] = YOLO(p)
                # 注释掉以下行，隐藏模型加载成功的提示
                # st.success(t('model_loaded').format(k=k, p=p))  # 现在k为Behavior，无中文
            except Exception as e:
                st.error(f"{t('model_loaded').split(':')[0]} {k} failed: {str(e)}")
    # 兜底逻辑：无任何模型加载成功时，尝试加载Ich
    if not models:
        if Path(WEIGHTS).exists():
            models["Ich"] = YOLO(WEIGHTS)
            st.warning(t('fallback_model'))
        else:
            st.error(t('no_available_model'))
    return models

MODELS = load_models()

# ====================== 核心工具函数 ======================
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
                rows.append({
                    t("category"): names.get(int(cls_np[i]), str(int(cls_np[i]))),
                    t("confidence"): float(conf_np[i]),
                    t("location"): [float(x) for x in xyxy_np[i].tolist()],
                })
        return pd.DataFrame(rows)
    if isinstance(res, list):
        rows = []
        for d in res or []:
            rows.append({
                t("category"): d.get("category") or d.get("class_name") or d.get("name") or d.get("cls"),
                t("confidence"): d.get("conf") or d.get("confidence"),
                t("location"): d.get("location") or d.get("bbox") or d.get("xyxy"),
                t("path"): d.get("path"),
            })
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
    # 修复KeyError：检查模型是否存在
    if model_key not in MODELS:
        default_model = list(MODELS.keys())[0] if MODELS else None
        st.warning(t('model_switch').format(k=model_key, default_model=default_model))
        model_key = "Ich" if "Ich" in MODELS else default_model
    if not model_key:
        raise RuntimeError(t('no_available_model'))
    r = MODELS[model_key].predict(source=pil_img, conf=c, imgsz=640, verbose=False)[0]
    im_bgr = r.plot()
    im_rgb = im_bgr[..., ::-1]
    vis_pil = Image.fromarray(im_rgb)
    df = detections_to_df(r)
    return vis_pil, df

# 原有视频处理函数
def process_video(video_bytes: bytes, model_key: str, conf: float | None = None, max_frames: int | None = None) -> Path:
    if not CV2_OK:
        raise RuntimeError(t("video_disabled"))
    in_path = Path("input_tmp.mp4");
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
        if not ok: break
        i += 1
        if max_frames and i > max_frames: break
        c = float(conf) if conf is not None else DEFAULT_CONF
        # 修复KeyError：检查模型是否存在
        if model_key not in MODELS:
            default_model = list(MODELS.keys())[0] if MODELS else None
            st.warning(t('model_switch').format(k=model_key, default_model=default_model))
            model_key = "Ich" if "Ich" in MODELS else default_model
        if not model_key:
            raise RuntimeError(t('no_available_model'))
        r = MODELS[model_key].predict(source=frame, conf=c, imgsz=640, verbose=False)[0]
        vw.write(r.plot())

    cap.release();
    vw.release()
    return out_path

# 轨迹分析核心函数
def calculate_fish_trajectory(video_bytes: bytes, model_key: str, conf: float = DEFAULT_CONF,
                              max_frames: int = None) -> dict:
    """
    分析视频中鱼类的运动轨迹（支持指定模型）
    :param model_key: 要使用的模型名称
    """
    if not CV2_OK:
        return {
            "success": False,
            "message": t("video_disabled"),
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": ""
        }

    # 校验模型是否存在
    if model_key not in MODELS:
        default_model = list(MODELS.keys())[0] if MODELS else None
        st.warning(t('model_switch').format(k=model_key, default_model=default_model))
        model_key = default_model
    if not model_key:
        return {
            "success": False,
            "message": t('no_available_model'),
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": ""
        }

    # 初始化变量
    prev_center = None
    total_distance = 0.0
    total_frames = 0
    trajectory_points = []

    # 动态匹配当前模型的类别名（适配不同模型）
    current_model = MODELS[model_key]
    model_class_names = current_model.names
    fish_keywords = ["healthy", "subhealthy", "diseased", "健康", "亚健康", "患病", "鱼", "fish"]
    fish_categories = set()
    for cls_idx, cls_name in model_class_names.items():
        if any(keyword.lower() in cls_name.lower() for keyword in fish_keywords):
            fish_categories.add(cls_name)
            fish_categories.add(t(cls_name))
    # 核心修改：将Healthy替换为health
    fish_categories.update({"健康", "亚健康", "患病", "health", "Subhealthy", "Diseased"})

    # 写入临时视频文件
    in_path = Path("traj_input_tmp.mp4")
    in_path.write_bytes(video_bytes)

    # 打开视频
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        return {
            "success": False,
            "message": t('cannot_read_video_file'),
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": ""
        }

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频写入器
    processed_video_path = Path(f"traj_processed_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(processed_video_path), fourcc, fps, (w, h))

    # 逐帧处理
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        if max_frames and total_frames > max_frames:
            break

        # 更新进度
        progress = min(total_frames / total_frames_total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{t('tracking_processing')} {total_frames}/{total_frames_total}")

        # 模型推理（使用指定的模型）
        try:
            r = MODELS[model_key].predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            cap.release()
            out.release()
            in_path.unlink(missing_ok=True)
            processed_video_path.unlink(missing_ok=True)
            return {
                "success": False,
                "message": f"{t('frame_inference_failed')} {str(e)}",
                "total_distance": 0,
                "average_speed": 0,
                "video_duration": 0,
                "total_frames": total_frames,
                "processed_video_path": ""
            }

        # 绘制检测框
        frame_with_detect = r.plot()

        # 提取当前帧鱼类中心坐标
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

        # 优化轨迹容错逻辑
        if current_center is not None:
            if prev_center is not None:
                distance = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])
                total_distance += distance
                cv2.line(frame_with_detect, prev_center, current_center, (0, 0, 255), 2)
            trajectory_points.append(current_center)
            cv2.circle(frame_with_detect, current_center, 5, (255, 0, 0), -1)
            prev_center = current_center

        # 写入视频帧
        out.write(frame_with_detect)

    # 清理资源
    cap.release()
    out.release()
    in_path.unlink(missing_ok=True)
    progress_bar.empty()
    status_text.empty()

    # 计算统计值
    video_duration = total_frames / fps if fps > 0 else 0
    average_speed = total_distance / video_duration if video_duration > 0 else 0

    # 优化提示信息
    if total_distance == 0:
        return {
            "success": True,
            "message": f"{t('no_fish_detected')} | {t('model_label')}{model_key} | {t('suggestions')}",
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": round(video_duration, 2),
            "total_frames": total_frames,
            "processed_video_path": str(processed_video_path) if processed_video_path.exists() else ""
        }

    return {
        "success": True,
        "message": f"{t('tracking_processing').replace('...', 'completed')} ({t('model_label')}{model_key})",
        "total_distance": round(total_distance, 2),
        "average_speed": round(average_speed, 2),
        "video_duration": round(video_duration, 2),
        "total_frames": total_frames,
        "processed_video_path": str(processed_video_path) if processed_video_path.exists() else ""
    }

def save_table_to_excel(df: pd.DataFrame, filename: str) -> Path:
    out = Path(filename).with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="detections" if st.session_state.language == 'zh' else "Detections", index=False)
    return out

def zip_files(files: list[Path], out_zip: Path) -> Path:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists(): zf.write(f, arcname=f.name)
    return out_zip

# ========= 模糊预测（核心修改：覆盖所有边界情况+异常处理） =========
@st.cache_resource
def build_fuzzy_sim():
    # 核心修改1：移除day/night，新增behavior（行为特征）
    behavior = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'behavior')  # 行为特征：1=健康，2=亚健康，3=患病
    surf = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'surf')          # 体表特征
    patho = ctrl.Antecedent(np.arange(1, 4.1, 0.1), 'patho')        # 病原存在性
    risk = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'risk')          # 风险结果

    # 行为特征隶属度函数（合并原日间+夜间逻辑）
    behavior['healthy'] = fuzz.trimf(behavior.universe, [1, 1, 1.5])    # 健康行为
    behavior['subhealthy'] = fuzz.trimf(behavior.universe, [1.5, 2, 2.5])# 亚健康行为
    behavior['diseased'] = fuzz.trimf(behavior.universe, [2.5, 3, 4])   # 患病行为

    # 体表特征隶属度函数（保留）
    surf['healthy'] = fuzz.trimf(surf.universe, [1, 1, 2])       # 体表健康
    surf['diseased'] = fuzz.trimf(surf.universe, [2, 3, 4])      # 体表异常

    # 病原存在性隶属度函数（保留）
    patho['absent'] = fuzz.trimf(patho.universe, [1, 1, 2])      # 无病原
    patho['present'] = fuzz.trimf(patho.universe, [2, 3, 4])     # 有病原

    # 风险结果隶属度函数（保留）
    risk['health'] = fuzz.trimf(risk.universe, [0, 1, 1.5])      # 健康
    risk['subhealth'] = fuzz.trimf(risk.universe, [1.5, 2, 2.5]) # 亚健康
    risk['diseased'] = fuzz.trimf(risk.universe, [2.5, 3, 4])    # 患病
    risk.defuzzify_method = 'centroid'  # 重心法解模糊

    # 核心修改2：重构模糊规则（覆盖所有边界情况）
    rules = [
        # 基础规则：行为健康+体表健康+无病原 → 健康
        ctrl.Rule(behavior['healthy'] & surf['healthy'] & patho['absent'], risk['health']),
        
        # 新增：行为健康+体表健康+病原存在 → 亚健康（关键修复）
        ctrl.Rule(behavior['healthy'] & surf['healthy'] & patho['present'], risk['subhealth']),
        
        # 高风险规则：行为患病 或 体表异常+有病原 → 患病
        ctrl.Rule(behavior['diseased'], risk['diseased']),
        ctrl.Rule(surf['diseased'] & patho['present'], risk['diseased']),
        
        # 中风险规则：行为亚健康 或 体表异常但无病原 → 亚健康
        ctrl.Rule(behavior['subhealthy'], risk['subhealth']),
        ctrl.Rule(surf['diseased'] & patho['absent'], risk['subhealth']),
        
        # 边界规则：行为健康但体表异常+有病原 → 亚健康（过渡状态）
        ctrl.Rule(behavior['healthy'] & surf['diseased'] & patho['present'], risk['subhealth']),
        
        # 边界规则：行为亚健康+体表健康+有病原 → 亚健康
        ctrl.Rule(behavior['subhealthy'] & surf['healthy'] & patho['present'], risk['subhealth']),
        
        # 边界规则：行为亚健康+体表异常+无病原 → 亚健康
        ctrl.Rule(behavior['subhealthy'] & surf['diseased'] & patho['absent'], risk['subhealth']),
        
        # 严格规则：行为患病+任何体表状态+有病原 → 高风险患病（权重提升）
        ctrl.Rule(behavior['diseased'] & patho['present'], risk['diseased']),
        
        # 补充：行为患病+体表异常+无病原 → 患病（覆盖最后边界）
        ctrl.Rule(behavior['diseased'] & surf['diseased'] & patho['absent'], risk['diseased']),
    ]
    
    # 调整规则权重（关键规则权重更高）
    for r in rules: r.weight = 1.0
    rules[3].weight = 2  # 行为患病规则权重加倍
    rules[4].weight = 2  # 体表异常+有病原规则权重加倍

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# 核心修改3：增加异常处理，确保所有情况都能返回结果
def fuzzy_predict(behavior_val: float, surf_val: float, patho_val: float) -> dict:
    try:
        sim = build_fuzzy_sim()
        sim.input['behavior'] = behavior_val  # 行为特征值
        sim.input['surf'] = surf_val          # 体表特征值
        sim.input['patho'] = patho_val        # 病原存在性值
        sim.compute()
        
        v = float(sim.output['risk'])
        # 结果映射（兼容中英文）
        if st.session_state.language == 'zh':
            status = "健康" if v < 1.5 else ("亚健康" if v < 2.5 else "患病")
        else:
            status = t("healthy") if v < 1.5 else (t("subhealthy") if v < 2.5 else t("diseased"))
        
        return {"risk_value": round(v, 1), "risk_status": status}
    
    except Exception as e:
        # 异常处理：返回默认值并提示
        st.warning(f"{t('fuzzy_calc_error')}{str(e)}")
        if st.session_state.language == 'zh':
            default_status = "亚健康"
        else:
            default_status = t("subhealthy")
        return {"risk_value": 2.0, "risk_status": default_status}

# ========================= 全局样式 =========================
st.markdown("""
<style>
.app-header {
  background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
  color: white; border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
  display:flex; align-items:center; gap:14px;
}
.app-title { font-size: 22px; font-weight: 700; letter-spacing:.3px; }
.app-subtitle { opacity:.9; font-size: 13px; }

.note {
  background:#EEF2FF; border:1px solid #E0E7FF; color:#3730A3;
  border-radius: 12px; padding: 10px 12px; margin: 6px 0 16px 0; font-size:13px;
}

.card {
  background: var(--secondary-bg, #F6F7FB);
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 14px;
  margin-bottom: 12px;
}

:root { --secondary-bg: #F6F7FB; }
[data-base-theme="light"] :root { --secondary-bg: #F6F7FB; }
[data-base-theme="dark"]  :root { --secondary-bg: #111827; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow:hidden; }

.stButton>button { border-radius: 10px; }
.block-container { padding-top: 0.6rem; padding-bottom: 1rem; }

.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #EEF2FF; color:#3730A3; border:1px solid #E0E7FF;
  padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight:600;
}

/* 语言切换按钮样式 */
.lang-switch {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 999;
}

/* 轨迹统计卡片样式 */
.traj-card {
  background: #f0f8ff;
  border: 1px solid #b8d4ff;
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
}
.traj-metric {
  font-size: 18px;
  font-weight: 600;
  color: #2563eb;
}
/* 健康程度卡片特殊样式 */
.health-card {
  background: #e8f4f8;
  border: 1px solid #4299e1;
}
.health-status {
  font-size: 20px;
  font-weight: 700;
  color: #2d3748;
}
.healthy { color: #48bb78 !important; }
.subhealthy { color: #ed8936 !important; }
.diseased { color: #e53e3e !important; }

/* 下拉选项样式优化 */
.stSelectbox > div > div {
  border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# 隐藏Streamlit默认组件
st.markdown("""
<style>
header[data-testid="stHeader"] {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"] .main .block-container { padding-top: 0.8rem !important; }
.app-header { margin-top: 4px; }
#svc-config { display: none !important; }
</style>
""", unsafe_allow_html=True)

# 语言切换按钮
with st.sidebar:
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button('中文', use_container_width=True):
            st.session_state.language = 'zh'
            st.rerun()
    with lang_col2:
        if st.button('English', use_container_width=True):
            st.session_state.language = 'en'
            st.rerun()

# 顶部导航条
st.markdown(f"""
<style>
.app-header {{
  background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
  color: white;
  border-radius: 14px;
  padding: 14px 18px;
  margin-bottom: 14px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
.app-title-row {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 8px;
}}
.app-icon {{
  font-size: 42px;
}}
.app-title {{
  font-size: 36px;
  font-weight: 800;
  letter-spacing: 1px;
}}
.app-subtitle {{
  font-size: 20px;
  opacity: 0.95;
}}
</style>
<div class="app-header">
  <div class="app-title-row">
    <div class="app-icon">🧪</div>
    <div class="app-title">{t('header_title')}</div>
  </div>
  <div class="app-subtitle">{t('header_subtitle')}</div>
</div>
""", unsafe_allow_html=True)

# 侧边栏（关键修改：模型选项映射Behavior）
with st.sidebar:
    st.markdown(f"### 🎓 {t('sidebar_university')}")
    st.markdown('<div id="svc-config">', unsafe_allow_html=True)
    base_url = "http://localhost:8080"
    ws_url_override = base_url.replace("http://", "ws://").replace("https://", "wss://")
    st.divider()
    st.header(t('sidebar_model'))
    # 仅显示已加载的模型（使用优化后的翻译名称）
    model_options = {
        "Ich": t("Ich"),
        "Tomont": t("Tomont"),
        "Behavior": t("Behavior"),
        "CiSurface": t("CiSurface"),
        "CiTomont": t("CiTomont"),
        "CroakerBehavior": t("CroakerBehavior"),
    }
    available_models = {k: model_options.get(k, k) for k in MODELS.keys()}
    if not available_models:
        st.error(t('no_available_model'))
        model_value = None
    else:
        default_model = "Ich" if "Ich" in available_models else list(available_models.keys())[0]
        model_value = st.selectbox(
            t('sidebar_model_type'),
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            index=list(available_models.keys()).index(default_model) if default_model in available_models else 0
        )
    # 显示当前模型（容错 + 优化显示）
    if model_value and model_value in MODELS:
        st.markdown(f"<span class='badge'>{t('sidebar_current_model')} <b>{available_models[model_value]}</b></span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge' style='color:red'>{t('no_available_model')}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= 标签页 =========================
tab_img, tab_folder, tab_video, tab_camera, tab_tracking, tab_fuzzy = st.tabs([
    t('tab_image'),
    t('tab_batch'),
    t('tab_video'),
    t('tab_camera'),
    t('tab_tracking'),
    t('tab_fuzzy')
])

# -------------------------------- 1) 图片检测 --------------------------------
with tab_img:
    st.markdown(f"#### {t('tab_image')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='card'><b>{t('image_original')}</b></div>", unsafe_allow_html=True)
        img_file = st.file_uploader(t('image_upload'), type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    key="single_img_main")
        if img_file:
            st.image(Image.open(img_file), caption=t('image_original'), use_column_width=True)
    with col2:
        st.markdown(f"<div class='card'><b>{t('image_detection')}</b></div>", unsafe_allow_html=True)
        run_single = st.button(t('image_run'), type="primary", use_container_width=True,
                               disabled=img_file is None or model_value is None)
        if run_single and img_file:
            with st.spinner(t('batch_processing').replace('：', '') if st.session_state.language == 'zh' else t('batch_processing')):
                det_img, df = predict_on_image(img_file.getvalue(), model_value)
            st.image(det_img, caption=t('image_result'), use_column_width=True)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(t('image_download_excel'), use_container_width=True):
                        xlsx_path = save_table_to_excel(df, "image_detect_result.xlsx")
                        st.download_button(t('image_download_excel'), data=open(xlsx_path, "rb").read(),
                                           file_name=xlsx_path.name,
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with c2:
                    if st.button(t('image_download_img'), use_container_width=True):
                        bio = io.BytesIO();
                        det_img.save(bio, format="JPEG")
                        st.download_button(t('image_download_img'), data=bio.getvalue(),
                                           file_name="image_detect_result.jpg",
                                           mime="image/jpeg")

# ----------------------------- 2) 批量图片检测 -----------------------------
with tab_folder:
    st.markdown(f"#### {t('tab_batch')}")
    files = st.file_uploader(
        t('batch_upload'),
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="multi_imgs",
    )
    go = st.button(t('batch_run'), type="primary", disabled=not files or model_value is None)
    if go and files:
        all_tables: List[pd.DataFrame] = []
        out_imgs: List[Path] = []
        progress = st.progress(0)
        status = st.empty()
        total = len(files)
        for i, f in enumerate(files, start=1):
            status.info(f"{t('batch_processing')}{f.name} ({i}/{total})")
            with st.spinner(f"{t('batch_processing')}{f.name}"):
                det_img, df = predict_on_image(f.getvalue(), model_value)
                if not df.empty:
                    df[t("path")] = f.name
                    all_tables.append(df)
                out_path = Path(f"{Path(f.name).stem}_detect.jpg")
                det_img.save(out_path)
                out_imgs.append(out_path)
            progress.progress(i / total)
        df_all = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
        if not df_all.empty:
            st.dataframe(df_all, use_container_width=True)
            xlsx_path = save_table_to_excel(df_all, "batch_detect.xlsx")
            st.download_button(
                t('batch_download_excel'),
                data=open(xlsx_path, "rb").read(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info(t('batch_no_results'))
        if out_imgs:
            zpath = zip_files(out_imgs, Path("batch_detect_images.zip"))
            st.download_button(
                t('batch_download_zip'),
                data=open(zpath, "rb").read(),
                file_name=zpath.name,
                mime="application/zip",
                use_container_width=True,
            )
        status.empty()
        progress.empty()

# -------------------------------- 3) 视频检测 --------------------------------
with tab_video:
    st.markdown(f"#### {t('tab_video')}")
    vid_file = st.file_uploader(
        t('video_upload'), type=["mp4", "mov", "avi", "mkv"], key="video_file"
    )
    run_vid = st.button(t('video_run'), type="primary",
                        disabled=(vid_file is None or not CV2_OK or model_value is None))
    if not CV2_OK:
        st.warning(t('video_disabled'))
    if run_vid and vid_file:
        with st.spinner(t('video_processing')):
            out_path = process_video(vid_file.getvalue(), model_value, max_frames=None)
        # 移除视频预览，仅保留下载按钮
        st.success(t('video_process_complete'))
        st.download_button(
            t('video_download'),
            data=open(out_path, "rb").read(),
            file_name=out_path.name,
            mime="video/mp4",
            use_container_width=True
        )

# -------------------------- 4) 摄像头检测 --------------------------
with tab_camera:
    st.markdown(f"#### {t('camera_title')}")
    st.caption(t('camera_caption'))
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False
    col_a, col_b = st.columns(2)
    if not st.session_state.cam_on:
        if col_a.button(t('camera_open'), type="primary"):
            st.session_state.cam_on = True
            st.rerun()
        col_b.button(t('camera_close'), disabled=True)
        st.info(t('camera_not_started'))
    else:
        if col_b.button(t('camera_close'), type="secondary"):
            st.session_state.cam_on = False
            st.rerun()
        col_a.button(t('camera_open'), disabled=True)
        snap = st.camera_input(t('camera_shot'), key="cam_shot")
        go = st.button(t('camera_detect'), type="primary", disabled=(snap is None or model_value is None))
        if go and snap is not None:
            with st.spinner(t('batch_processing').replace('：', '') if st.session_state.language == 'zh' else t('batch_processing')):
                det_img, df = predict_on_image(snap.getvalue(), model_value)
            st.image(det_img, caption=t('image_result'), use_column_width=True)
            if not df.empty:
                st.dataframe(df, use_container_width=True)

# -------------------------- 5) 轨迹分析 --------------------------
with tab_tracking:
    # 添加小图标 + 更新标题文本
    st.markdown(f"#### 🐠 {t('tracking_title')}")

    # 移除专属模型选择，复用左侧全局model_value
    if model_value and model_value in MODELS:
        st.markdown(f"<div class='note'>{t('sidebar_current_model')} {available_models[model_value]}</div>", unsafe_allow_html=True)
    else:
        st.error(t('no_available_model'))

    # 视频上传
    vid_file = st.file_uploader(
        t('tracking_upload'),
        type=["mp4", "mov", "avi", "mkv"],
        key="tracking_video_file",
        help=t('video_format_help')
    )

    # 展示原始视频（保留原始视频预览，仅移除处理后的视频预览）
    if vid_file:
        st.markdown(f"### 🎬 {t('original_video')}")
        st.video(vid_file)

    # 置信度阈值（降低默认值）
    conf_threshold = st.slider(
        t('conf_threshold'),
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="tracking_conf",
        help=t('conf_threshold_help')
    )

    # 最大分析帧数限制
    max_frames = st.number_input(
        t('max_frames'),
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        key="tracking_max_frames",
        help=t('max_frames_help')
    )

    # 日间/夜间选择（在开始按钮上方）
    time_period = st.selectbox(
        t("time_period"),
        options=[t("daytime"), t("nighttime")],
        index=0,
        key="tracking_time_period",
        help=t("time_period") + " - " + t('suggestions').split(':')[0]
    )

    # 开始分析按钮（禁用条件改为全局model_value）
    run_tracking = st.button(
        t('tracking_run'),
        type="primary",
        disabled=(vid_file is None or not CV2_OK or model_value is None),
        use_container_width=True
    )

    # CV2未加载提示
    if not CV2_OK:
        st.warning(t('video_disabled'))

    # 执行轨迹分析（使用全局model_value）
    if run_tracking and vid_file and CV2_OK and model_value:
        with st.spinner(t('tracking_processing')):
            result = calculate_fish_trajectory(
                video_bytes=vid_file.getvalue(),
                model_key=model_value,
                conf=conf_threshold,
                max_frames=max_frames if max_frames > 0 else None
            )

        # 展示结果
        st.markdown(f"### 📊 {t('analysis_results')}")
        if result["success"]:
            # 计算健康程度
            health_status = get_health_status(result["average_speed"], time_period)
            
            # 成功结果展示
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div class="traj-card">
                    <p>{t('total_distance')}</p>
                    <p class="traj-metric">{result['total_distance']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="traj-card">
                    <p>{t('average_speed')}</p>
                    <p class="traj-metric">{result['average_speed']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="traj-card">
                    <p>{t('video_duration')}</p>
                    <p class="traj-metric">{result['video_duration']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="traj-card">
                    <p>{t('total_frames')}</p>
                    <p class="traj-metric">{result['total_frames']}</p>
                </div>
                """, unsafe_allow_html=True)

            # 健康程度展示（特殊样式）
            with col5:
                # 根据健康状态添加不同样式类
                status_class = ""
                if health_status in [t("healthy"), "健康", "Healthy"]:
                    status_class = "healthy"
                elif health_status in [t("subhealthy"), "亚健康", "Subhealthy"]:
                    status_class = "subhealthy"
                else:
                    status_class = "diseased"
                
                st.markdown(f"""
                <div class="traj-card health-card">
                    <p>{t('health_status')}</p>
                    <p class="traj-metric health-status {status_class}">{health_status}</p>
                </div>
                """, unsafe_allow_html=True)

            # 移除处理后视频预览，仅保留下载按钮
            if result["processed_video_path"] and Path(result["processed_video_path"]).exists():
                st.markdown(f"### 📥 {t('processed_video_download')}")
                # 下载按钮（文件名包含优化后的模型名）
                st.download_button(
                    label=t('download_traj_video'),
                    data=open(result["processed_video_path"], "rb").read(),
                    file_name=f"traj_video_{available_models[model_value]}_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            # 提示信息
            if result["total_distance"] == 0:
                st.info(result["message"])
            else:
                st.success(f"{result['message']} | {t('health_status_label')}{health_status}")

        else:
            # 失败提示
            st.error(f"{t('analysis_failed')}{result['message']}")

# -------------------------------- 6) 模糊预测（核心修改：合并行为特征） --------------------------------
with tab_fuzzy:
    st.markdown(f"#### {t('fuzzy_title')}")
    st.markdown(f"<div class='card'><b>{t('fuzzy_input')}</b></div>", unsafe_allow_html=True)
    
    # 核心修改：定义新的选项映射（仅保留行为特征、体表特征、病原存在性）
    behavior_options = {
        t('healthy'): 1.0,      # 健康行为
        t('subhealthy'): 2.0,   # 亚健康行为
        t('diseased'): 3.0      # 患病行为
    }
    surface_options = {
        t('healthy'): 1.0,      # 体表健康
        t('diseased'): 3.0      # 体表异常
    }
    pathogen_options = {
        t('pathogen_absent'): 1.0,  # 无病原
        t('pathogen_present'): 3.0  # 有病原
    }
    
    # 布局调整：改为三列布局（行为特征、体表特征、病原存在性）
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 行为特征下拉框（合并原日间+夜间）
        behavior_label = st.selectbox(
            t('fuzzy_behavior'),
            options=list(behavior_options.keys()),
            index=0,  # 默认选中"健康"
            key="behavior_feature"
        )
        behavior_val = behavior_options[behavior_label]
    
    with col2:
        # 体表特征下拉框（保留）
        surface_label = st.selectbox(
            t('fuzzy_surface'),
            options=list(surface_options.keys()),
            index=0,  # 默认选中"健康"
            key="surface_feature"
        )
        surface_val = surface_options[surface_label]
    
    with col3:
        # 病原存在性下拉框（保留）
        pathogen_label = st.selectbox(
            t('fuzzy_pathogen'),
            options=list(pathogen_options.keys()),
            index=1,  # 默认选中"存在"
            key="pathogen_feature"
        )
        pathogen_val = pathogen_options[pathogen_label]
    
    # 预测按钮
    if st.button(t('fuzzy_predict'), type="primary", use_container_width=True):
        # 核心修改：调用模糊预测函数（仅传3个参数）
        r = fuzzy_predict(behavior_val, surface_val, pathogen_val)
        st.success(t('fuzzy_result').format(risk_value=r['risk_value'], risk_status=r['risk_status']))




