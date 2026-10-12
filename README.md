# 🎭 Facial Emotion Recognition & Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-CNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

<p align="center">
  <b>Giving machines the ability to recognize facial expressions — one frame at a time.</b>
  <br/>
  A deep learning project that detects faces and classifies visible facial expressions using a Convolutional Neural Network (CNN).
</p>

---

## ✨ About the Project

What if a computer could recognize a smile, a frown, or a surprised expression?

**Facial Emotion Recognition & Detection** combines deep learning and computer vision to identify faces in images and predict the facial expression shown. The project uses a CNN trained on the FER-2013 dataset and provides an interactive interface built with Streamlit.

> 🧠 Built to explore how computer vision and deep learning turn image pixels into predictions.

## 🚀 Features

- 🖼️ **Image Recognition** — Upload an image to analyze a detected face.
- 📸 **Camera Input** — Capture an image through the web interface.
- 🎥 **Live Webcam Detection** — Run facial emotion predictions through a webcam.
- 🤖 **CNN-Based Classification** — Predict one of seven emotion categories.
- 📊 **Prediction Scores** — View confidence scores for the available classes.
- 🌐 **Interactive Interface** — Explore the model through a Streamlit web app.

## 🎭 Emotions Detected

| Emotion | Label |
|---|---|
| 😠 | Angry |
| 🤢 | Disgust |
| 😨 | Fear |
| 😄 | Happy |
| 😐 | Neutral |
| 😢 | Sad |
| 😲 | Surprise |

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and training |
| OpenCV | Face detection and image processing |
| NumPy | Numerical operations |
| Pillow | Image handling |
| Streamlit | Interactive web application |
| FER-2013 | Facial expression image dataset |

## 🧠 How It Works

```text
       📷 Input Image / Camera
                 |
                 v
       🔍 Face Detection (OpenCV)
                 |
                 v
       🖼️ Grayscale & Resize
              (48 × 48)
                 |
                 v
       🤖 Trained CNN Model
                 |
                 v
       📊 Emotion Probabilities
                 |
                 v
       🎭 Predicted Expression
```

**Note:** This is an experimental model and its performance can vary across emotion categories, lighting conditions, face angles, and image quality. The accuracy is a baseline, not production-level performance.

## 📂 Project Structure

```text
facial-emotion-recognition/
│
├── 📁 models/              # Trained models (not tracked by Git)
├── 📁 sample_images/       # Example images for testing
├── 📁 src/
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── model.py            # CNN architecture
│   ├── train.py            # Model training and evaluation
│   ├── predict_image.py    # Image-based prediction
│   └── webcam.py           # Webcam-based prediction
│
├── app.py                  # Streamlit web application
├── check_setup.py          # Environment verification
├── requirements.txt        # Python dependencies
├── .gitignore              # Excluded files and folders
└── README.md               # Project documentation
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rautsamiksha0311/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the FER-2013 dataset from Kaggle:

🔗 [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

Arrange it in the following structure:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...same seven emotion folders
```

### 5. Train the model

```bash
python -m src.train
```

The training script saves model checkpoints under `models/`. Ensure the trained model file expected by the application is available before running it.

### 6. Launch the application

```bash
python -m streamlit run app.py
```

Your browser should open the local Streamlit application.

## 🔮 Future Improvements

- 📈 Improve accuracy through hyperparameter tuning and data augmentation.
- 📊 Add a confusion matrix and per-class precision, recall, and F1-score.
- 🧪 Compare different CNN architectures.
- ⚡ Optimize inference speed and face detection.
- ☁️ Deploy the application online.

## ⚠️ Limitations & Responsible Use

- Facial expressions do not reliably reveal a person's true emotional state.
- Predictions may be affected by lighting, occlusion, face angle, and image quality.
- The model may perform differently across people and demographic groups.
- This project is intended for learning and experimentation, not for consequential decisions about people.

📧 **Email:** [rautsamiksha0311@gmail.com](mailto:rautsamiksha0311@gmail.com)

🐙 **GitHub:** [@rautsamiksha0311](https://github.com/rautsamiksha0311)

---

<p align="center">
  💜 <b>Built with Python, deep learning, and curiosity.</b>
  <br/>
  <i>Exploring the intersection of AI and computer vision.</i>
</p>
