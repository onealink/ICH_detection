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

# ====================== 语言配置（新增轨迹跟踪翻译） ======================
if 'language' not in st.session_state:
    st.session_state.language = 'zh'  # 默认中文

# 翻译字典（新增轨迹跟踪相关字段）
translations = {
    'zh': {
        # 原有翻译保留，新增以下字段
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
        # 原有翻译
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
        'fuzzy_day': '日间行为（1~3）',
        'fuzzy_night': '夜间行为（1~3）',
        'fuzzy_surface': '体表特征（1~3）',
        'fuzzy_pathogen': '病原特征（1~3）',
        'fuzzy_predict': '🧪 预测',
        'fuzzy_result': '风险值: {risk_value}，状态: {risk_status}',
        '行为': '行为分析模型',
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
        # 原有翻译保留，新增以下字段
        'tab_tracking': '📍 Trajectory Tracking',
        'tracking_title': 'Goldfish Motion Trajectory Analysis',
        'tracking_upload': 'Upload Video File',
        'tracking_run': '🚀 Start Trajectory Analysis',
        'tracking_processing': 'Analyzing video trajectory...',
        'total_distance': 'Total Distance (pixels)',
        'average_speed': 'Average Movement Speed (pixels/sec)',
        'video_duration': 'Video Duration (sec)',
        'total_frames': 'Total Frames',
        'no_fish_detected': 'No goldfish detected, cannot calculate trajectory data',
        # 原有翻译
        'page_title': 'YOLO Disease Detection',
        'header_title': 'Fish Parasitic Disease Detection',
        'header_subtitle': 'Image / Batch / Video / Camera / Trajectory Tracking / Fuzzy Prediction — One-stop Detection Platform',
        'sidebar_university': 'Ningbo University · Disease Laboratory',
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
        'video_processing': 'Local video processing... (May be slow depending on CPU)',
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
        'fuzzy_day': 'Day Behavior (1~3)',
        'fuzzy_night': 'Night Behavior (1~3)',
        'fuzzy_surface': 'Surface Features (1~3)',
        'fuzzy_pathogen': 'Pathogen Features (1~3)',
        'fuzzy_predict': '🧪 Predict',
        'fuzzy_result': 'Risk Value: {risk_value}, Status: {risk_status}',
        '行为': 'Behavior Analysis',
        'Ich': 'Ichthyophthirius Disease',
        'Tomont': 'Tomont',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'category': 'Category',
        'confidence': 'Confidence',
        'location': 'Location',
        'path': 'Path',
        'model_not_found': 'Model file not found: {p} ({k} model)',
        'model_loaded': 'Successfully loaded model: {k} -> {p}'
    }
}

# 获取当前语言翻译
def t(key):
    return translations[st.session_state.language].get(key, key)

# ====================== 页面配置 ======================
st.set_page_config(page_title=t('page_title'), page_icon="🧪", layout="wide")

# ====================== 模型加载 ======================
BASE_DIR = Path(__file__).parent
WEIGHTS = BASE_DIR / "best.pt"  # Ich模型
TOMONT_WEIGHTS = BASE_DIR / "tomont.best.pt"  # 新增Tomont模型路径
# 新增：行为模型路径（guijibest.pt，与best.pt同目录）
BEHAVIOR_WEIGHTS = BASE_DIR / "guijibest.pt"  # 行为分析模型
IMG_DIR = BASE_DIR / "img"
# 新增：将行为模型加入路径字典
MODEL_PATHS = {"Ich": str(WEIGHTS), "Tomont": str(TOMONT_WEIGHTS), "行为": str(BEHAVIOR_WEIGHTS)}  # 移除Lyc，分别对应不同模型
DEFAULT_CONF = 0.6  # 默认置信度

@st.cache_resource
def load_models():
    models = {}
    for k, p in MODEL_PATHS.items():
        if not Path(p).exists():
            st.error(t('model_not_found').format(p=p, k=k))
        else:
            models[k] = YOLO(p)
            st.success(t('model_loaded').format(k=k, p=p))
    return models

MODELS = load_models()

# ====================== 核心工具函数 ======================
def detections_to_df(res) -> pd.DataFrame:
    if hasattr(res, "boxes") and hasattr(res, "names"):
        rows = []
        names = getattr(res, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_np  = boxes.cls.detach().cpu().numpy().astype(int)
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
    r = MODELS[model_key].predict(source=pil_img, conf=c, imgsz=640, verbose=False)[0]
    im_bgr = r.plot()
    im_rgb = im_bgr[..., ::-1]
    vis_pil = Image.fromarray(im_rgb)
    df = detections_to_df(r)
    return vis_pil, df

# 原有视频处理函数（保留，供视频检测标签页使用）
def process_video(video_bytes: bytes, model_key: str, conf: float | None = None, max_frames: int | None = None) -> Path:
    if not CV2_OK:
        raise RuntimeError(t("video_disabled"))
    in_path = Path("input_tmp.mp4"); in_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened(): raise RuntimeError("无法读取视频" if st.session_state.language == 'zh' else "Cannot read video")

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
        r = MODELS[model_key].predict(source=frame, conf=c, imgsz=640, verbose=False)[0]
        vw.write(r.plot())

    cap.release(); vw.release()
    return out_path

# 新增：轨迹分析核心函数（修改后）
def calculate_fish_trajectory(video_bytes: bytes, conf: float = DEFAULT_CONF, max_frames: int = None) -> dict:
    """
    分析视频中金鱼的运动轨迹（强制使用Ich模型），返回统计结果+带轨迹的视频路径
    返回值：{
        "total_distance": 总路程(像素),
        "average_speed": 平均速度(像素/秒),
        "video_duration": 视频时长(秒),
        "total_frames": 总帧数,
        "processed_video_path": 带轨迹的视频路径,
        "success": 是否成功,
        "message": 提示信息
    }
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
    
    # 初始化变量
    prev_center = None  # 上一帧金鱼中心坐标
    total_distance = 0.0  # 总路程
    total_frames = 0  # 总帧数
    trajectory_points = []  # 轨迹坐标列表（用于绘制）
    # 定义金鱼类别（排除包囊）
    fish_categories = {t("healthy"), t("subhealthy"), t("diseased"), "健康", "亚健康", "患病", "Healthy", "Subhealthy", "Diseased"}
    
    # 写入临时视频文件
    in_path = Path("traj_input_tmp.mp4")
    in_path.write_bytes(video_bytes)
    
    # 打开视频
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        return {
            "success": False,
            "message": "无法读取视频文件" if st.session_state.language == 'zh' else "Cannot read video file",
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": 0,
            "total_frames": 0,
            "processed_video_path": ""
        }
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # 视频帧率
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    
    # 初始化视频写入器（保存带轨迹的视频）
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
        
        # 限制最大帧数（防止超长视频卡顿）
        if max_frames and total_frames > max_frames:
            break
        
        # 更新进度
        progress = min(total_frames / total_frames_total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{t('tracking_processing')} {total_frames}/{total_frames_total}")
        
        # 模型推理（强制使用Ich模型）
        try:
            r = MODELS["Ich"].predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            cap.release()
            out.release()
            in_path.unlink(missing_ok=True)
            processed_video_path.unlink(missing_ok=True)
            return {
                "success": False,
                "message": f"帧推理失败: {str(e)}",
                "total_distance": 0,
                "average_speed": 0,
                "video_duration": 0,
                "total_frames": total_frames,
                "processed_video_path": ""
            }
        
        # 绘制检测框（保留原检测效果）
        frame_with_detect = r.plot()
        
        # 提取当前帧金鱼的中心坐标（取置信度最高的）
        current_center = None
        max_conf = 0.0
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                cls_name = r.names.get(cls_idx, "")
                # 过滤出金鱼类别
                if cls_name in fish_categories:
                    conf_score = float(box.conf.item())
                    if conf_score > max_conf:
                        max_conf = conf_score
                        # 计算检测框中心坐标
                        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        current_center = (center_x, center_y)
        
        # 计算与上一帧的距离 + 记录轨迹点
        if prev_center is not None and current_center is not None:
            # 欧氏距离公式：√[(x2-x1)² + (y2-y1)²]
            distance = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])
            total_distance += distance
            # 添加轨迹点
            trajectory_points.append(current_center)
            # 绘制轨迹连线（红色，线宽2）
            cv2.line(frame_with_detect, prev_center, current_center, (0, 0, 255), 2)
        # 绘制当前中心坐标点（蓝色，半径5）
        if current_center is not None:
            cv2.circle(frame_with_detect, current_center, 5, (255, 0, 0), -1)
            trajectory_points.append(current_center)
        
        # 更新上一帧坐标
        if current_center is not None:
            prev_center = current_center
        
        # 写入带轨迹的视频帧
        out.write(frame_with_detect)
    
    # 清理资源
    cap.release()
    out.release()
    in_path.unlink(missing_ok=True)
    progress_bar.empty()
    status_text.empty()
    
    # 计算视频时长和平均速度
    video_duration = total_frames / fps if fps > 0 else 0
    average_speed = total_distance / video_duration if video_duration > 0 else 0
    
    # 检测是否有有效轨迹
    if total_distance == 0:
        return {
            "success": True,
            "message": t("no_fish_detected"),
            "total_distance": 0,
            "average_speed": 0,
            "video_duration": round(video_duration, 2),
            "total_frames": total_frames,
            "processed_video_path": str(processed_video_path) if processed_video_path.exists() else ""
        }
    
    return {
        "success": True,
        "message": "轨迹分析完成",
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

# ========= 模糊预测 =========
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
    for r in rules: r.weight = 1.0
    rules[4].weight = 2; rules[5].weight = 2

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def fuzzy_predict(day_val: float, night_val: float, surf_val: float, patho_val: float) -> dict:
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

# 侧边栏
with st.sidebar:
    st.markdown(f"### 🎓 {t('sidebar_university')}")
    st.markdown('<div id="svc-config">', unsafe_allow_html=True)
    base_url = "http://localhost:8080"
    ws_url_override = base_url.replace("http://", "ws://").replace("https://", "wss://")
    st.divider()
    st.header(t('sidebar_model'))
    model_options = {"Ich": t('Ich'), "Tomont": t('Tomont'), "行为": t('行为')}
    model_value = st.selectbox(t('sidebar_model_type'), options=list(model_options.keys()),
                           format_func=lambda x: f"{x}（{model_options[x]}）")
    st.markdown(f"<span class='badge'>{t('sidebar_current_model')} <b>{model_value}</b></span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= 标签页（新增轨迹跟踪标签） =========================
# 修改标签页定义，加入轨迹跟踪
tab_img, tab_folder, tab_video, tab_camera, tab_tracking, tab_fuzzy = st.tabs([
    t('tab_image'), 
    t('tab_batch'), 
    t('tab_video'), 
    t('tab_camera'), 
    t('tab_tracking'),  # 新增：轨迹跟踪标签页
    t('tab_fuzzy')
])

# -------------------------------- 1) 图片检测 --------------------------------
with tab_img:
    st.markdown(f"#### {t('tab_image')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='card'><b>{t('image_original')}</b></div>", unsafe_allow_html=True)
        img_file = st.file_uploader(t('image_upload'), type=["jpg","jpeg","png","bmp","webp"], key="single_img_main")
        if img_file:
            st.image(Image.open(img_file), caption=t('image_original'), use_column_width=True)
    with col2:
        st.markdown(f"<div class='card'><b>{t('image_detection')}</b></div>", unsafe_allow_html=True)
        run_single = st.button(t('image_run'), type="primary", use_container_width=True, disabled=img_file is None)
        if run_single and img_file:
            with st.spinner("本地模型推理中..." if st.session_state.language == 'zh' else "Local model inferencing..."):
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
                        st.download_button(t('image_download_img'), data=bio.getvalue(), file_name="image_detect_result.jpg",
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
    go = st.button(t('batch_run'), type="primary", disabled=not files)
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
    run_vid = st.button(t('video_run'), type="primary", disabled=(vid_file is None or not CV2_OK))
    if not CV2_OK:
        st.warning(t('video_disabled'))
    if run_vid and vid_file:
        with st.spinner(t('video_processing')):
            out_path = process_video(vid_file.getvalue(), model_value, max_frames=None)
        st.video(str(out_path))
        st.download_button(
            t('video_download'),
            data=open(out_path, "rb").read(),
            file_name=out_path.name,
            mime="video/mp4",
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
        go = st.button(t('camera_detect'), type="primary", disabled=(snap is None))
        if go and snap is not None:
            with st.spinner("本地模型推理中..." if st.session_state.language == 'zh' else "Local model inferencing..."):
                det_img, df = predict_on_image(snap.getvalue(), model_value)
            st.image(det_img, caption=t('image_result'), use_column_width=True)
            if not df.empty:
                st.dataframe(df, use_container_width=True)

# -------------------------- 5) 轨迹跟踪（新增） --------------------------
with tab_tracking:
    st.markdown(f"#### {t('tracking_title')}")
    # 新增：轨迹跟踪模型选择
    model_tracking = st.selectbox("选择分析模型", options=["Ich", "行为"], format_func=lambda x: t(x))
    st.markdown(f"<div class='note'>当前使用模型：{t(model_tracking)}（{model_tracking}）</div>", unsafe_allow_html=True)

# 再将推理处改为：
r = MODELS[model_tracking].predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
    
    # 视频上传
    vid_file = st.file_uploader(
        t('tracking_upload'),
        type=["mp4", "mov", "avi", "mkv"],
        key="tracking_video_file",
        help="支持常见视频格式，建议时长不超过1分钟以保证分析速度"
    )
    
    # 展示原始视频（上传后立即显示）
    if vid_file:
        st.markdown("### 🎬 原始视频")
        st.video(vid_file)
    
    # 置信度阈值设置
    conf_threshold = st.slider(
        "检测置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONF,
        step=0.05,
        key="tracking_conf"
    )
    
    # 最大帧数限制（防止超长视频卡顿）
    max_frames = st.number_input(
        "最大分析帧数（0=无限制）",
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        key="tracking_max_frames"
    )
    
    # 开始分析按钮
    run_tracking = st.button(
        t('tracking_run'),
        type="primary",
        disabled=(vid_file is None or not CV2_OK),
        use_container_width=True
    )
    
    # CV2未加载提示
    if not CV2_OK:
        st.warning(t('video_disabled'))
    
        # 执行轨迹分析
    if run_tracking and vid_file and CV2_OK:
        with st.spinner(t('tracking_processing')):
            # 调用轨迹分析函数（强制使用Ich模型，不再传model_value）
            result = calculate_fish_trajectory(
                video_bytes=vid_file.getvalue(),
                conf=conf_threshold,
                max_frames=max_frames if max_frames > 0 else None
            )
        
        # 展示结果
        st.markdown("### 📊 分析结果")
        if result["success"]:
            # 成功结果展示
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            # 新增：展示带轨迹的检测视频
            if result["processed_video_path"] and Path(result["processed_video_path"]).exists():
                st.markdown("### 🎬 带轨迹的检测视频")
                st.video(result["processed_video_path"])
                # 新增：下载带轨迹视频的按钮
                st.download_button(
                    label="下载带轨迹的检测视频",
                    data=open(result["processed_video_path"], "rb").read(),
                    file_name=f"traj_video_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            # 提示信息
            if result["total_distance"] == 0:
                st.info(result["message"])
            else:
                st.success("轨迹分析完成！")
        
        else:
            # 失败提示
            st.error(f"分析失败：{result['message']}")

# -------------------------------- 6) 模糊预测 --------------------------------
with tab_fuzzy:
    st.markdown(f"#### {t('fuzzy_title')}")
    st.markdown(f"<div class='card'><b>{t('fuzzy_input')}</b></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        day_behavior   = st.number_input(t('fuzzy_day'),  min_value=1.0, max_value=3.0, value=3.0, step=1.0)
        night_behavior = st.number_input(t('fuzzy_night'),  min_value=1.0, max_value=3.0, value=1.0, step=1.0)
    with c2:
        surface_features = st.number_input(t('fuzzy_surface'), min_value=1.0, max_value=3.0, value=3.0, step=1.0)
        pathogen         = st.number_input(t('fuzzy_pathogen'), min_value=1.0, max_value=3.0, value=3.0, step=1.0)
    if st.button(t('fuzzy_predict'), type="primary"):
        r = fuzzy_predict(day_behavior, night_behavior, surface_features, pathogen)
        st.success(t('fuzzy_result').format(risk_value=r['risk_value'], risk_status=r['risk_status']))



