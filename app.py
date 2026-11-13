import os
import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadGuard AI | Pothole Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #080b10;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0d1117;
}

.hero {
    background: linear-gradient(135deg, #151a22, #0d1117);
    border: 1px solid #2a3340;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    margin: 0;
    color: white;
}

.hero-title span {
    color: #f59e0b;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-top: 10px;
    line-height: 1.6;
}

.card {
    background: #111720;
    border: 1px solid #273140;
    border-radius: 16px;
    padding: 20px;
}

.card-title {
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.card-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

.metric {
    background: #111720;
    border: 1px solid #273140;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: 900;
    margin-top: 5px;
}

.warning {
    background: #2a1717;
    border: 1px solid #713333;
    border-radius: 15px;
    padding: 18px;
    margin: 20px 0;
}

.success-box {
    background: #10251a;
    border: 1px solid #285c3a;
    border-radius: 15px;
    padding: 18px;
    margin: 20px 0;
}

.section-title {
    color: white;
    font-size: 24px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    border-top: 1px solid #273140;
    margin-top: 40px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("RoadGuard AI")
    st.caption("Pothole Intelligence System")

    st.divider()

    st.subheader("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        0.95,
        0.35,
        0.05
    )

    st.divider()

    st.subheader("AI Model")

    st.write("**Model:** YOLO")
    st.write("**Task:** Object Detection")
    st.write("**Target:** Road Potholes")
    st.write("**Mode:** Local Inference")

    st.divider()

    if model is not None:
        st.success("Model Ready")
    else:
        st.error("Model Missing")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Road<span>Guard</span> AI
</div>

<div class="hero-subtitle">
AI-powered road inspection system that detects potholes,
measures detection confidence and provides visual road-damage
intelligence using a trained YOLO computer-vision model.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error("Trained model not found.")

    st.info(
        f"Place your trained model here:\n\n"
        f"`{MODEL_PATH}`"
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Road Image Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a road image and the trained AI model will detect "
    "potholes automatically."
)

uploaded_file = st.file_uploader(
    "Choose road image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# MAIN
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.error(f"Unable to read image: {e}")
        st.stop()


    # --------------------------------------------------------
    # IMAGE INFO
    # --------------------------------------------------------

    width, height = image.size

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Image Width", f"{width}px")

    with c2:
        st.metric("Image Height", f"{height}px")

    with c3:
        st.metric(
            "File Size",
            f"{uploaded_file.size / 1024:.1f} KB"
        )


    st.divider()


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "ANALYZE ROAD IMAGE",
        type="primary",
        use_container_width=True
    )


    if analyze:

        with st.spinner("RoadGuard AI is analyzing the image..."):

            start_time = time.time()

            try:

                results = model.predict(
                    source=np.array(image),
                    conf=confidence,
                    verbose=False
                )

                inference_time = time.time() - start_time

                result = results[0]

            except Exception as e:

                st.error(f"Detection failed: {e}")
                st.stop()


            # ------------------------------------------------
            # DETECTIONS
            # ------------------------------------------------

            detections = []

            if result.boxes is not None:

                for i in range(len(result.boxes)):

                    box = result.boxes[i]

                    conf = float(
                        box.conf.cpu().numpy()
                    )

                    cls_id = int(
                        box.cls.cpu().numpy()
                    )

                    coords = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    x1, y1, x2, y2 = coords

                    class_name = model.names.get(
                        cls_id,
                        str(cls_id)
                    )

                    detections.append({
                        "id": i + 1,
                        "class": class_name,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2)
                    })


            # ------------------------------------------------
            # SAVE SESSION RESULT
            # ------------------------------------------------

            st.session_state["result"] = {
                "image": image,
                "result": result,
                "detections": detections,
                "time": inference_time
            }


# ============================================================
# DISPLAY RESULT
# ============================================================

if "result" in st.session_state:

    data = st.session_state["result"]

    result = data["result"]
    detections = data["detections"]
    inference_time = data["time"]


    total = len(detections)


    if total > 0:

        highest = max(
            d["confidence"]
            for d in detections
        )

        average = sum(
            d["confidence"]
            for d in detections
        ) / total

    else:

        highest = 0
        average = 0


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if total > 0:

        st.markdown(
            f"""
            <div class="warning">
                <h3 style="color:#ff6b6b;">
                {total} POTHOLE(S) DETECTED
                </h3>

                <p style="color:#cbd5e1;">
                Road damage has been detected by the AI model.
                Review the highlighted areas below.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="success-box">
                <h3 style="color:#4ade80;">
                NO POTHOLES DETECTED
                </h3>

                <p style="color:#cbd5e1;">
                No potholes passed the selected confidence threshold.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Detection Summary</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Potholes Found",
            total
        )

    with m2:
        st.metric(
            "Highest Confidence",
            f"{highest * 100:.1f}%"
        )

    with m3:
        st.metric(
            "Average Confidence",
            f"{average * 100:.1f}%"
        )

    with m4:
        st.metric(
            "Inference Time",
            f"{inference_time * 1000:.1f} ms"
        )


    # --------------------------------------------------------
    # VISUAL RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Visual Inspection</div>',
        unsafe_allow_html=True
    )

    original_col, detected_col = st.columns(2)


    with original_col:

        st.subheader("Original Image")

        st.image(
            data["image"],
            use_container_width=True
        )


    with detected_col:

        st.subheader("AI Detection Result")

        annotated = result.plot()

        st.image(
            annotated,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DETECTION DETAILS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Detection Intelligence</div>',
        unsafe_allow_html=True
    )


    if detections:

        for d in detections:

            x1, y1, x2, y2 = d["box"]

            with st.container(border=True):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"**Detection #{d['id']}**"
                    )

                with col2:
                    st.write(
                        f"**Class:** {d['class']}"
                    )

                with col3:
                    st.write(
                        f"**Confidence:** "
                        f"{d['confidence'] * 100:.1f}%"
                    )

                st.caption(
                    f"Bounding Box: "
                    f"({x1}, {y1}) → ({x2}, {y2})"
                )

    else:

        st.info(
            "No detection records available."
        )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⬇Export Result</div>',
        unsafe_allow_html=True
    )

    annotated = result.plot()

    from PIL import Image as PILImage
    import io

    output = PILImage.fromarray(annotated)

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        "⬇Download Annotated Image",
        data=buffer.getvalue(),
        file_name=(
            "RoadGuard_Pothole_Result_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ),
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# ABOUT
# ============================================================

st.markdown(
    '<div class="section-title">About RoadGuard AI</div>',
    unsafe_allow_html=True
)

a1, a2, a3 = st.columns(3)


with a1:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    AI Detection
    </div>

    <div class="card-text">
    A trained YOLO object-detection model identifies
    potholes from road images.
    </div>

    </div>
    """, unsafe_allow_html=True)


with a2:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    Spatial Intelligence
    </div>

    <div class="card-text">
    Every detected pothole contains its bounding-box
    coordinates and confidence score.
    </div>

    </div>
    """, unsafe_allow_html=True)


with a3:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    Local Inference
    </div>

    <div class="card-text">
    Images are processed locally using the trained
    YOLO model without an external prediction API.
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>RoadGuard AI</b> — Road Pothole Detection System

<br><br>

</div>
""", unsafe_allow_html=True)