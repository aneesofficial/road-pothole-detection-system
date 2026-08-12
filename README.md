# 🚧 RoadGuard AI — Intelligent Road Pothole Detection System

> **AI-powered Computer Vision system for automatic pothole detection from road images using YOLO Object Detection and Streamlit.**

RoadGuard AI is an end-to-end computer vision project designed to identify potholes in road images using a trained **YOLO object detection model**.

The system takes a road image as input, detects potholes, draws bounding boxes around detected road damage, calculates confidence scores, and presents the results through an interactive **Streamlit web application**.

---

## 🎯 Project Objective

Road damage, especially potholes, can create serious problems for:

* 🚗 Vehicle safety
* 🛣️ Road infrastructure
* 🚑 Emergency transportation
* 💰 Road maintenance planning
* 👷 Infrastructure inspection

Manual road inspection is time-consuming and difficult to scale.

**RoadGuard AI** provides an automated computer-vision solution that can inspect road images and identify potholes quickly.

---

# ✨ Key Features

### 🤖 AI-Based Pothole Detection

Uses a trained YOLO object detection model to identify potholes from road images.

### 📦 Bounding Box Detection

Every detected pothole is highlighted with a bounding box.

### 🎯 Confidence Score

The system displays the confidence level of every detection.

### ⚙️ Adjustable Detection Threshold

Users can control the confidence threshold directly from the Streamlit sidebar.

### 📊 Detection Intelligence

The application provides:

* Total potholes detected
* Highest confidence
* Average confidence
* Inference time
* Bounding-box coordinates

### 🖼️ Original vs AI Result

The application displays the original road image alongside the AI-generated detection result.

### ⬇️ Result Download

Users can download the annotated pothole-detection image.

### ⚡ Local AI Inference

The trained model runs locally without requiring an external prediction API.

### 🌐 Interactive Web Application

A professional Streamlit interface provides an easy way to test the trained model.

---

# 🧠 Technology Stack

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Core programming language          |
| YOLO         | Object detection                   |
| Ultralytics  | YOLO implementation                |
| OpenCV       | Image processing                   |
| NumPy        | Numerical image operations         |
| Pillow       | Image loading and conversion       |
| Streamlit    | Web application                    |
| Google Colab | Model training and experimentation |
| VS Code      | Development environment            |

---

# 📂 Project Structure

```text
Road-Pothole-Detection-System/
│
├── app.py
│
├── models/
│   └── best.pt
│
├── outputs/
│
├── README.md
│
└── requirements.txt
```

### Important Files

**`app.py`**

Main Streamlit web application responsible for:

* Image upload
* YOLO model loading
* Detection
* Confidence calculation
* Result visualization
* Result downloading

**`models/best.pt`**

The trained YOLO pothole detection model.

**`outputs/`**

Directory used for generated/output files.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

---

# 📊 Dataset

The project uses an annotated pothole image dataset.

Dataset statistics used during preparation:

```text
Total Images       : 665
Total XML Annotations: 665
Detection Class    : pothole
```

Example annotation:

```text
Image: potholes536.png
Width: 400
Height: 400

Objects: 2

Class: pothole
Bounding Box: 144 197 239 258

Class: pothole
Bounding Box: 18 112 58 137
```

The original annotations were converted into a YOLO-compatible object-detection format for model training.

---

# 🔄 Computer Vision Pipeline

The complete project follows this pipeline:

```text
Road Images
     ↓
Image Annotation
     ↓
Annotation Conversion
     ↓
YOLO Dataset Preparation
     ↓
Train / Validation / Test Split
     ↓
YOLO Model Training
     ↓
Best Model Selection
     ↓
Test Prediction
     ↓
Streamlit Web Application
     ↓
Pothole Detection
     ↓
Bounding Boxes + Confidence
     ↓
Infrastructure Inspection Result
```

---

# 🤖 YOLO Model

The trained model is stored as:

```text
models/best.pt
```

The web application loads the trained model instead of retraining it.

Conceptually:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
```

The model then performs inference on uploaded road images.

---

# 🧪 Model Testing

The trained model was tested on:

```text
Test Images: 67
```

The model successfully generated detections across the test images.

Example prediction outputs included:

```text
1 pothole
2 potholes
3 potholes
4 potholes
5 potholes
7 potholes
17 potholes
21 potholes
```

The prediction pipeline completed successfully and generated result images.

Average reported inference performance during testing was approximately:

```text
Preprocessing : 2.1 ms
Inference     : 14.8 ms
Postprocessing: 1.1 ms
```

> Inference speed can vary depending on hardware, image resolution and execution environment.

---

# 🌐 Streamlit Web Application

RoadGuard AI provides an interactive web interface for real-world model testing.

### Application Workflow

```text
1. Open Web Application
          ↓
2. Upload Road Image
          ↓
3. Select Confidence Threshold
          ↓
4. Analyze Image
          ↓
5. YOLO Model Performs Detection
          ↓
6. Potholes Are Highlighted
          ↓
7. Detection Statistics Are Displayed
          ↓
8. Annotated Result Can Be Downloaded
```

---

# 📊 Web App Output

After analyzing an image, the application provides:

### Detection Summary

```text
Potholes Found
Highest Confidence
Average Confidence
Inference Time
```

### Visual Results

```text
Original Road Image
        +
AI Detection Result
```

### Detection Details

For every detected pothole:

```text
Detection ID
Class
Confidence
Bounding Box Coordinates
```

Example:

```text
Detection #1
Class: pothole
Confidence: 91.4%

Bounding Box:
(144, 197) → (239, 258)
```

---

# ⚙️ Installation

## 1. Clone or open the project

Open the project folder in VS Code.

```bash
cd Road-Pothole-Detection-System
```

---

## 2. Create Conda Environment

```bash
conda create -n pothole_ai python=3.11 -y
```

Activate it:

```bash
conda activate pothole_ai
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If required packages are not installed, install the main dependencies:

```bash
pip install streamlit ultralytics opencv-python pillow numpy
```

---

# ▶️ Run the Web Application

From the project root directory:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 📷 Using the Application

1. Start the Streamlit application.
2. Upload a road image.
3. Adjust the confidence threshold if required.
4. Click **Analyze Road Image**.
5. Wait for YOLO inference.
6. Review detected potholes.
7. Check confidence scores and bounding boxes.
8. Compare original and detected images.
9. Download the annotated result.

---

# 🎛️ Confidence Threshold

The application provides a configurable confidence threshold.

For example:

```text
0.20 → More detections, potentially more false positives

0.35 → Balanced detection

0.50 → More conservative detections

0.70+ → Only highly confident detections
```

The ideal threshold depends on the intended deployment scenario.

---

# 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │     User Image      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Streamlit Interface │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   YOLO best.pt      │
                │  Trained Detector   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Pothole Detection   │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Bounding Boxes              Confidence
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ Detection Dashboard │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Download Result     │
                └─────────────────────┘
```

---

# 🔐 Privacy

Road images are processed locally by the application.

The application does not require an external cloud prediction API for inference.

---

# 🚀 Future Improvements

RoadGuard AI can be extended into a larger intelligent infrastructure platform.

### 📍 GPS-Based Pothole Mapping

Store detected potholes with GPS coordinates and visualize them on a map.

### 🗺️ Road Damage Heatmap

Generate a geographical heatmap showing areas with high pothole density.

### 📱 Mobile Integration

Connect the model with a mobile application for field inspection.

### 🎥 Real-Time Detection

Extend the system from static images to video and live camera streams.

### 📈 Road Condition Scoring

Develop a road-condition score based on:

* Number of potholes
* Pothole size
* Detection confidence
* Pothole density

### 🗃️ Inspection History

Store previous inspections and compare road conditions over time.

### 🏛️ Infrastructure Management

Integrate the system into a dashboard for municipal road-maintenance teams.

---

# ⚠️ Current Limitations

* The current model focuses on pothole detection.
* Detection performance depends on image quality.
* Very small or heavily occluded potholes may be difficult to detect.
* Confidence threshold affects the number of detections.
* Real-world deployment would require testing on diverse road conditions.
* GPS-based mapping and automatic maintenance prioritization are not currently implemented.

---

# 📌 Project Highlights

```text
665        Annotated Images
67         Test Images
1          Detection Class
YOLO       Object Detection
Streamlit  Web Application
OpenCV     Image Processing
Local      AI Inference
```

---

# 👨‍💻 Project

**RoadGuard AI — Intelligent Road Pothole Detection System**

Built as a Computer Vision / AI Engineering project using:

```text
Python
YOLO
Ultralytics
OpenCV
NumPy
Pillow
Streamlit
```

---

# 🏁 Conclusion

RoadGuard AI demonstrates how computer vision can be applied to automate road infrastructure inspection.

The system combines a trained YOLO object-detection model with a professional Streamlit interface to transform road images into useful pothole-detection information.

Instead of manually inspecting every road image, the system can automatically identify potholes, locate them using bounding boxes, provide confidence scores, measure inference performance, and generate downloadable inspection results.

The project establishes a foundation for a larger **AI-powered road infrastructure monitoring and maintenance platform**.
