
from pathlib import Path
import html
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

EMOTION_STYLES = {
    "Angry": {"icon": "😠", "color": "#ef4444"},
    "Disgust": {"icon": "🤢", "color": "#84cc16"},
    "Fear": {"icon": "😨", "color": "#a78bfa"},
    "Happy": {"icon": "😊", "color": "#fbbf24"},
    "Neutral": {"icon": "😐", "color": "#94a3b8"},
    "Sad": {"icon": "😢", "color": "#60a5fa"},
    "Surprise": {"icon": "😲", "color": "#f472b6"},
}

st.set_page_config(
    page_title="Facial Emotion AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111827 55%, #172033);
    color: #f8fafc;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border: 1px solid #334155;
    border-radius: 22px;
    background: linear-gradient(120deg, #172554, #1e293b);
    margin-bottom: 22px;
}
.hero h1 {
    color: #f8fafc;
    font-size: 2.4rem;
    margin: 0 0 8px 0;
}
.hero p {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 0;
}
div[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #64748b;
    border-radius: 14px;
    padding: 16px;
}
div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
}
.stButton button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 44px;
}
.result-card {
    background: #172554;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 12px;
}
.footer {
    color: #94a3b8;
    font-size: 0.85rem;
    text-align: center;
    padding-top: 24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Facial Emotion AI</h1>
    <p>Computer vision · Deep learning · Facial expression analysis</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Analyze a face from an uploaded image or a webcam snapshot."
)

@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")

    return model, detector

def analyze_image(image):
    model, detector = load_resources()
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    result = rgb.copy()
    predictions = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32) / 255.0
        face = face.reshape(1, 48, 48, 1)

        scores = model.predict(face, verbose=0)[0]
        index = int(np.argmax(scores))
        emotion = EMOTIONS[index]
        confidence = float(scores[index]) * 100

        cv2.rectangle(
            result, (x, y), (x + w, y + h), (34, 197, 94), 2
        )
        cv2.putText(
            result,
            f"{emotion.title()}: {confidence:.1f}%",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (34, 197, 94),
            2
        )

        predictions.append({
            "emotion": emotion.title(),
            "confidence": confidence
        })

    return result, predictions

def show_results(image, predictions):
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Detection Preview")
        st.image(
            image,
            caption="Face detection output",
            use_container_width=True
        )

    with right:
        st.markdown("### Prediction Results")

        if not predictions:
            st.warning(
                "No face detected. Try a clear, front-facing image "
                "with good lighting."
            )
            return

        for i, prediction in enumerate(predictions, start=1):
            emotion = prediction["emotion"]
            confidence = prediction["confidence"]
            style = EMOTION_STYLES.get(
                emotion,
                {"icon": "🙂", "color": "#60a5fa"}
            )
            safe_emotion = html.escape(emotion)

            st.markdown(
                f"""
                <div class="result-card"
                     style="border:1px solid {style['color']};
                            border-left:6px solid {style['color']};">
                    <div style="color:#bfdbfe;font-size:0.85rem;">
                        FACE {i}
                    </div>
                    <div style="color:{style['color']};
                                font-size:1.8rem;font-weight:700;">
                        {style['icon']} {safe_emotion}
                    </div>
                    <div style="color:#cbd5e1;margin-top:6px;">
                        Model score: {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

        st.caption(
            "The score reflects the model's prediction, "
            "not a verified emotional state."
        )

st.divider()

mode = st.radio(
    "Choose input method",
    ["Upload Image", "Webcam Snapshot"],
    horizontal=True
)

st.divider()

try:
    if mode == "Upload Image":
        st.markdown("### Upload an Image")
        st.write("Choose a JPG or PNG containing a visible face.")

        uploaded = st.file_uploader(
            "Select image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Original image", width=350)

            if st.button(
                "Analyze Facial Expression",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing image..."):
                    result, predictions = analyze_image(image)

                show_results(result, predictions)

    else:
        st.markdown("### Webcam Snapshot")
        st.write(
            "Allow camera access in your browser, then capture a frame."
        )

        captured = st.camera_input("Capture your face")

        if captured is not None:
            image = Image.open(captured).convert("RGB")

            with st.spinner("Analyzing captured frame..."):
                result, predictions = analyze_image(image)

            show_results(result, predictions)

except Exception as error:
    st.error(f"Unable to analyze the image: {error}")
    st.caption("Check the VS Code terminal for more details.")

st.divider()

st.markdown("""
<div class="footer">
    FACIAL EMOTION AI
    <br>
    Built with Python · TensorFlow · OpenCV · Streamlit
    <br><br>
    Predictions describe visible facial-expression categories,
    not verified emotions. Current test accuracy: 51.71%.
</div>
""", unsafe_allow_html=True)
