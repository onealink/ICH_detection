import streamlit as st
# 移除未使用的导入：from streamlit_extras.switch_page_button import switch_page
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
# ... 其余代码保持不变
# 可选导入 OpenCV（云端没有 GUI，用 headless 轮子即可；失败时禁用视频）
try:
    import cv2  # noqa: F401
    CV2_OK = True
except Exception:
    CV2_OK = False
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

# ====================== 语言配置 ======================
# 初始化语言状态
if 'language' not in st.session_state:
    st.session_state.language = 'zh'  # 默认中文

# 翻译字典
translations = {
    'zh': {
        'page_title': 'YOLO病害检测',
        'header_title': '鱼类寄生虫病检测',
        'header_subtitle': '图片 / 批量 / 视频 / 摄像头 / 模糊预测 — 一站式检测台',
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
        'Ich': '多子小瓜虫病',
        'Tomont': '包囊',
        'healthy': '健康',
        'subhealthy': '亚健康',
        'diseased': '患病',
        'category': '类别',
        'confidence': '置信度',
        'location': '位置',
        'path': '路径'
    },
    'en': {
        'page_title': 'YOLO Disease Detection',
        'header_title': 'Fish Parasitic Disease Detection',
        'header_subtitle': 'Image / Batch / Video / Camera / Fuzzy Prediction — One-stop Detection Platform',
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
        'Ich': 'Ichthyophthirius Disease',
        'Tomont': 'Tomont',
        'healthy': 'Healthy',
        'subhealthy': 'Subhealthy',
        'diseased': 'Diseased',
        'category': 'Category',
        'confidence': 'Confidence',
        'location': 'Location',
        'path': 'Path'
    }
}

# 获取当前语言翻译
def t(key):
    return translations[st.session_state.language].get(key, key)

# ====================== 页面配置 ======================
st.set_page_config(page_title=t('page_title'), page_icon="🧪", layout="wide")

# ====================== 原有代码（修改文本为翻译调用） ======================

# 以当前文件所在目录为基准
BASE_DIR = Path(__file__).parent
WEIGHTS = BASE_DIR / "best.pt"  # Ich模型
TOMONT_WEIGHTS = BASE_DIR / "tomont.best.pt"  # 新增Tomont模型路径
IMG_DIR = BASE_DIR / "img"
MODEL_PATHS = {"Ich": str(WEIGHTS), "Tomont": str(TOMONT_WEIGHTS)}  # 移除Lyc，分别对应不同模型
DEFAULT_CONF = 0.6  # 默认置信度

# 你的模型清单（可扩展多个）
# ========= 本地模型与工具 =========

# 如果三个类别共用同一权重，先都指向 best.pt；将来有不同权重再改这里的路径即可
# MODEL_PATHS = {"Lyc": "best.pt", "Ich": "best.pt", "Tomont": "tomont.best.pt"}

@st.cache_resource
def load_models():
    models = {}
    for k, p in MODEL_PATHS.items():
        if not Path(p).exists():
            st.error(f"模型文件不存在：{p}（{k}模型）")  # 新增：验证文件存在
        models[k] = YOLO(p)
        st.success(f"成功加载模型：{k} -> {p}")  # 新增：打印加载日志
    return models

MODELS = load_models()

def detections_to_df(res) -> pd.DataFrame:
    """
    统一转表：
    - Ultralytics Results（单帧）对象：从 res.boxes 提 cls/conf/xyxy。
    - 老接口 list[dict]：继续使用 d.get(...) 兼容。
    """
    # A) Ultralytics Results 对象
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

    # B) 老的 list[dict] 结构（保持兼容，如果你之后不用，可以删掉这段）
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

    # 已是 DataFrame 直接返回；其它类型给空表
    if isinstance(res, pd.DataFrame):
        return res

    return pd.DataFrame()

def predict_on_image(img_input, model_key: str, conf: float | None = None):
    # 统一转 PIL
    if isinstance(img_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, Image.Image):
        pil_img = img_input.convert("RGB")
    elif isinstance(img_input, (str, Path)):
        pil_img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:
            pil_img = Image.fromarray(img_input)  # 灰度
        elif img_input.ndim == 3:
            if CV2_OK:
                pil_img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            else:
                # 假设 BGR -> RGB（无 cv2 时用通道反转）
                pil_img = Image.fromarray(img_input[..., ::-1])
        else:
            raise TypeError(f"Unsupported numpy shape: {img_input.shape}")
    else:
        raise TypeError(f"Unsupported type: {type(img_input)}")

    # 推理
    c = float(conf) if conf is not None else DEFAULT_CONF
    # r = MODELS[model_key].predict(source=pil_img, conf=float(conf), imgsz=640, verbose=False)[0]
    r = MODELS[model_key].predict(source=pil_img, conf=c, imgsz=640, verbose=False)[0]

    # 可视化（Ultralytics 返回 BGR ndarray）
    im_bgr = r.plot()
    im_rgb = im_bgr[..., ::-1]  # 不依赖 cv2
    vis_pil = Image.fromarray(im_rgb)

    df = detections_to_df(r)
    return vis_pil, df

def process_video(video_bytes: bytes, model_key: str, conf: float | None = None, max_frames: int | None = None) -> Path:
    if not CV2_OK:
        raise RuntimeError(t("video_disabled"))
    """逐帧推理并输出 mp4，返回输出视频路径"""
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
        # r = MODELS[model_key].predict(source=frame, conf=float(conf), imgsz=640, verbose=False)[0]
        r = MODELS[model_key].predict(source=frame, conf=c, imgsz=640, verbose=False)[0]
        vw.write(r.plot())

    cap.release(); vw.release()
    return out_path

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

# ========= 模糊预测（和你后端一致的 scikit-fuzzy 规则） =========
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

# ========================= 全局设置 & 主题扩展 =========================

# 统一的 CSS：导航条 / 卡片 / 标签 / 表格 / 按钮
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
</style>
""", unsafe_allow_html=True)

# 隐藏 Streamlit 顶部栏（方案 B）
st.markdown("""
<style>
header[data-testid="stHeader"] {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"] .main .block-container { padding-top: 0.8rem !important; }
.app-header { margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# 只隐藏我们自定义的服务配置容器（更稳）
st.markdown("""
<style>
#svc-config { display: none !important; }  /* ← 被隐藏的容器 */
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
/* ===== 顶部横幅整体样式 ===== */
.app-header {{
  background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
  color: white;
  border-radius: 14px;     /* 圆角稍微小一些 */
  padding: 14px 18px;      /* 原来 26x20，缩小到更紧凑 */
  margin-bottom: 14px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}


/* ===== 图标与标题一行 ===== */
.app-title-row {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;  /* 图标与标题间距 */
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

# ------------------------- 侧边栏 -------------------------
with st.sidebar:
    # ======= 这里放你的校徽 / 项目简介（会显示）=======
    # 把 school_logo.png 放到同级目录后取消下一行注释即可：
    # st.image("school_logo.png", use_container_width=True)
    st.markdown(f"""
    ### 🎓 {t('sidebar_university')}
    """)
    # st.image("img/img1.png", width='stretch')
    # st.image(str(IMG_DIR / "img1.png"), use_column_width=True)

    # st.markdown("---")
    # ======= 以下为“服务配置 + 模型参数”区域，外面包了一个容器，已通过 CSS 隐藏 =======
    st.markdown('<div id="svc-config">', unsafe_allow_html=True)

    # st.header("⚙️ 服务配置")
    # base_url = st.text_input("后端地址", value="http://localhost:8080",
    #                          help="示例：http://127.0.0.1:8000 或 http://localhost:8080")
    # default_ws = base_url.replace("http://", "ws://").replace("https://", "wss://")
    # ws_url_override = st.text_input("WebSocket 基地址（可选）", value=default_ws,
    #                                 help="通常与后端一致，自动从后端地址推导")

    base_url = "http://localhost:8080"
    ws_url_override = base_url.replace("http://", "ws://").replace("https://", "wss://")
    st.divider()
    st.header(t('sidebar_model'))
    model_options = {"Ich": t('Ich'), "Tomont": t('Tomont')}
    model_value = st.selectbox(t('sidebar_model_type'), options=list(model_options.keys()),
                               format_func=lambda x: f"{x}（{model_options[x]}）")
    # conf = st.slider("置信度阈值", 0.05, 1.0, 0.6, 0.05)
    st.markdown(f"<span class='badge'>{t('sidebar_current_model')} <b>{model_value}</b></span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # ← 结束隐藏容器

# # 顶部提示条
# st.markdown("""
# <div class="note">
# 💡 小提示：左侧可展示校徽与项目简介；内部“服务配置/模型参数”已隐藏但仍生效。如需临时显示，可把 CSS 中的 #svc-config 隐藏规则去掉。
# </div>
# """, unsafe_allow_html=True)

# ========================= 工具函数 =========================
def b64_to_pil(maybe_b64):
    """兼容纯 base64 / data URL / bytes；URL 返回 None 交由 st.image(url)。"""
    import io, base64
    from PIL import Image
    if maybe_b64 is None:
        raise ValueError("empty image input" if st.session_state.language == 'en' else "空图片输入")
    if isinstance(maybe_b64, str) and maybe_b64.strip().lower().startswith(("http://", "https://")):
        return None
    if isinstance(maybe_b64, (bytes, bytearray)):
        return Image.open(io.BytesIO(maybe_b64)).convert("RGB")
    if isinstance(maybe_b64, str):
        s = maybe_b64.strip()
        if s.lower().startswith("data:image"):
            parts = s.split(",", 1)
            s = parts[1] if len(parts) > 1 else ""
        s = s.replace("\n", "").replace("\r", "").replace(" ", "")
        missing = len(s) % 4
        if missing:
            s += "=" * (4 - missing)
        raw = base64.b64decode(s, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise TypeError(f"unsupported type for image: {type(maybe_b64)}" if st.session_state.language == 'en' else f"不支持的图片类型: {type(maybe_b64)}")

def ensure_ok(resp: requests.Response):
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")

def save_table_to_excel(df: pd.DataFrame, filename: str) -> Path:
    out = Path(filename).with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="detections" if st.session_state.language == 'en' else "检测结果", index=False)
    return out

def zip_files(files: List[Path], out_zip: Path) -> Path:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                zf.write(f, arcname=f.name)
    return out_zip

# ========================= 标签页 =========================
tab_img, tab_folder, tab_video, tab_camera, tab_fuzzy = st.tabs([
    t('tab_image'), 
    t('tab_batch'), 
    t('tab_video'), 
    t('tab_camera'), 
    t('tab_fuzzy')
])

# -------------------------------- 1) 图片检测 --------------------------------
with tab_img:
    st.markdown(f"#### {t('tab_image')}")
    col1, col2 = st.columns(2)

    # 左侧：原图
    with col1:
        st.markdown(f"<div class='card'><b>{t('image_original')}</b></div>", unsafe_allow_html=True)
        img_file = st.file_uploader(t('image_upload'), type=["jpg","jpeg","png","bmp","webp"], key="single_img_main")
        if img_file:
            st.image(Image.open(img_file), caption=t('image_original'), use_column_width=True)

    # 右侧：检测与结果
    with col2:
        st.markdown(f"<div class='card'><b>{t('image_detection')}</b></div>", unsafe_allow_html=True)
        run_single = st.button(t('image_run'), type="primary", use_container_width=True, disabled=img_file is None)

        if run_single and img_file:
            files = {"file": (img_file.name, img_file.getvalue(), img_file.type or "image/jpeg")}
            data = {"model_type": model_value}
            # params = {"conf": conf}

            with st.spinner("本地模型推理中..." if st.session_state.language == 'zh' else "Local model inferencing..."):
                # det_img, df = predict_on_image(img_file.getvalue(), model_value, conf)
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
                # det_img, df = predict_on_image(f.getvalue(), model_value, conf)
                det_img, df = predict_on_image(f.getvalue(), model_value)

                # 结果表
                if not df.empty:
                    df[t("path")] = f.name
                    all_tables.append(df)

                # 保存标注图到本地，稍后打包下载
                out_path = Path(f"{Path(f.name).stem}_detect.jpg")
                det_img.save(out_path)
                out_imgs.append(out_path)

            progress.progress(i / total)

        # 汇总表格
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

        # 打包标注图
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
    # run_vid = st.button("🚀 开始视频检测", type="primary", disabled=vid_file is None)
    run_vid = st.button(t('video_run'), type="primary", disabled=(vid_file is None or not CV2_OK))
    if not CV2_OK:
        st.warning(t('video_disabled'))

    if run_vid and vid_file:
        with st.spinner(t('video_processing')):
            # 本地逐帧推理并导出处理后的视频
            out_path = process_video(
                # vid_file.getvalue(), model_value, conf, max_frames=None
                vid_file.getvalue(), model_value, max_frames=None
            )
        st.video(str(out_path))
        st.download_button(
            t('video_download'),
            data=open(out_path, "rb").read(),
            file_name=out_path.name,
            mime="video/mp4",
        )

# -------------------------- 4) 摄像头检测（拍照版，手动开启） --------------------------
with tab_camera:
    st.markdown(f"#### {t('camera_title')}")
    st.caption(t('camera_caption'))

    # 初始化状态
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

        # 只有在 cam_on=True 时才渲染 camera_input，避免页面加载就触发权限与取流
        snap = st.camera_input(t('camera_shot'), key="cam_shot")

        go = st.button(t('camera_detect'), type="primary", disabled=(snap is None))
        if go and snap is not None:
            with st.spinner("本地模型推理中..." if st.session_state.language == 'zh' else "Local model inferencing..."):
                # det_img, df = predict_on_image(snap.getvalue(), model_value, conf)
                det_img, df = predict_on_image(snap.getvalue(), model_value)
            st.image(det_img, caption=t('image_result'), use_column_width=True)
            if not df.empty:
                st.dataframe(df, use_container_width=True)

# -------------------------------- 5) 模糊预测 --------------------------------
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




