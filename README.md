# Detector de PPE con YOLOv8

Detector de Equipo de Protección Personal usando **YOLOv8** y **Streamlit**.
Tarea para el curso de Inteligencia Artificial Avanzada — UNAB Digital.

## Clases detectadas (7)

`boots` · `earmuffs` · `glasses` · `gloves` · `helmet` · `person` · `vest`

## Cómo correrlo localmente

```bash
git clone https://github.com/leydymf/reconocimiento-de-imagenes-PPE-.git
cd reconocimiento-de-imagenes-PPE-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> El archivo del modelo entrenado (`best.pt`) viene incluido en el repo.

## Modos de uso

La app tiene 3 pestañas:

- **Imagen** — subir un archivo JPG/PNG.
- **Foto de cámara** — tomar una foto con la webcam.
- **Webcam en vivo** — stream en tiempo real con detección por frame.

En la barra lateral se ajustan los umbrales de **confianza** e **IoU** en vivo.

## Re-entrenar el modelo

El cuaderno en [`notebooks/cuaderno_entrenamiento_ppe.ipynb`](notebooks/cuaderno_entrenamiento_ppe.ipynb) contiene el pipeline completo para entrenar en Google Colab:

1. Descarga el dataset **PPE Factory** desde Roboflow.
2. Entrena YOLOv8n con transfer learning (50 épocas, `imgsz=640`).
3. Copia `best.pt` automáticamente a Google Drive al terminar.

> Requiere una API key de Roboflow — crea una gratis en [roboflow.com](https://roboflow.com).

## Métricas del modelo

Entrenado sobre **9,770 imágenes** del dataset PPE Factory:

| Métrica    | Valor  |
|------------|--------|
| mAP50      | 0.76   |
| mAP50-95   | 0.51   |
| Precisión  | 0.76   |
| Recall     | 0.71   |

## Stack

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — modelo de detección.
- [Streamlit](https://streamlit.io/) — interfaz web.
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) — streaming de webcam.
- [Roboflow](https://roboflow.com/) — dataset.
