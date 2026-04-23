"""App Streamlit para detección de PPE con YOLOv8.

Detecta 7 clases: boots, earmuffs, glasses, gloves, helmet, person, vest.

Modos disponibles:
- Imagen: subir un archivo JPG/PNG.
- Foto: tomar una foto con la cámara.
- Webcam en vivo: stream continuo con detección en tiempo real.
"""

from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_COLORS = {
    "boots": (255, 128, 0),
    "earmuffs": (255, 0, 255),
    "glasses": (0, 255, 255),
    "gloves": (255, 255, 0),
    "helmet": (0, 200, 0),
    "person": (200, 200, 200),
    "vest": (0, 100, 255),
}


@st.cache_resource(show_spinner="Cargando modelo YOLOv8...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def detections_to_dataframe(result, class_names) -> pd.DataFrame:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["clase", "confianza", "x1", "y1", "x2", "y2"])
    rows = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        rows.append({
            "clase": class_names[cls_id],
            "confianza": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return pd.DataFrame(rows).sort_values("confianza", ascending=False).reset_index(drop=True)


def run_detection(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float):
    results = model.predict(image_bgr, conf=conf, iou=iou, verbose=False)
    result = results[0]
    annotated_bgr = result.plot()
    df = detections_to_dataframe(result, model.names)
    return annotated_bgr, df


st.set_page_config(page_title="Detector PPE", page_icon="🦺", layout="wide")

st.title("🦺 Detector de PPE con YOLOv8")
st.caption(
    "Detecta equipo de protección personal: botas, orejeras, gafas, guantes, casco, persona, chaleco."
)

with st.sidebar:
    st.header("⚙️ Configuración")
    conf_thr = st.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05)
    iou_thr = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.markdown(
        "**Modelo:** `best.pt` debe estar en la misma carpeta que `app.py`.\n\n"
        "Descárgalo desde el cuaderno de Colab al terminar el entrenamiento."
    )

if not MODEL_PATH.exists():
    st.error(
        f"No se encontró **{MODEL_PATH.name}** en `{MODEL_PATH.parent}`.\n\n"
        "1. Corre el cuaderno `cuaderno_entrenamiento_ppe.ipynb` en Colab.\n"
        "2. Descarga el archivo `best.pt` al finalizar.\n"
        "3. Cópialo en esta carpeta y recarga la página."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

with st.expander("Clases que reconoce el modelo"):
    st.write(", ".join(sorted(model.names.values())))

tab_img, tab_snap, tab_live = st.tabs(["📷 Imagen", "📸 Foto de cámara", "🎥 Webcam en vivo"])

with tab_img:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        annotated_bgr, df = run_detection(model, img_bgr, conf_thr, iou_thr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.subheader(f"Detecciones ({len(df)})")
            st.image(annotated_rgb, use_container_width=True)

        if not df.empty:
            st.subheader("Resumen")
            counts = df["clase"].value_counts().rename_axis("clase").reset_index(name="total")
            st.dataframe(counts, use_container_width=True, hide_index=True)
            st.subheader("Detalle")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron objetos. Prueba bajando la confianza en la barra lateral.")

with tab_snap:
    snap = st.camera_input("Toma una foto con tu cámara")
    if snap is not None:
        pil_img = Image.open(snap).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        annotated_bgr, df = run_detection(model, img_bgr, conf_thr, iou_thr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.subheader(f"Detecciones ({len(df)})")
        st.image(annotated_rgb, use_container_width=True)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.conf = 0.25
        self.iou = 0.45

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.model is not None:
            results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
            img = results[0].plot()
        return av.VideoFrame.from_ndarray(img, format="bgr24")


with tab_live:
    st.markdown(
        "Activa la cámara para ver la detección en tiempo real. "
        "Ajusta la confianza y el IoU desde la barra lateral mientras el stream esté activo."
    )

    ctx = webrtc_streamer(
        key="ppe-live",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor is not None:
        ctx.video_processor.model = model
        ctx.video_processor.conf = conf_thr
        ctx.video_processor.iou = iou_thr
