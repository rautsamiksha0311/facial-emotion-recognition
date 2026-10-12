
# Facial Emotion Recognition and Detection

A deep learning application that detects faces and classifies visible facial expressions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

## Features

- Facial expression classification into seven categories
- Face detection using OpenCV
- Image upload and analysis
- Webcam snapshot analysis
- CNN model built and trained using TensorFlow/Keras
- Interactive web interface built with Streamlit
- Prediction scores for detected expressions

## Emotion Categories

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Technology Stack

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Pillow
- Streamlit
- FER-2013 dataset

## Model Architecture

The project uses a Convolutional Neural Network with convolutional layers, batch normalization, max pooling, dropout, and dense layers.

Input images are converted to grayscale, resized to 48 × 48 pixels, and normalized before prediction.

## Model Performance

The initial CNN achieved **51.71% test accuracy** on the FER-2013 test split.

This is a baseline result. Predictions can be incorrect, particularly for visually similar expressions or images with challenging lighting.

## Project Structure

```text
facial-emotion-recognition/
├── dataset/
├── models/
├── sample_images/
├── results/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── predict_image.py
│   └── webcam.py
├── app.py
├── check_setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository.
2. Open the project directory.
3. Create and activate a Python virtual environment.
4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Ensure the trained model is available at `models/best_model.keras`.
6. Start the application:

```bash
python -m streamlit run app.py
```

7. Open the local URL displayed in the terminal.

## Dataset

The project uses FER-2013 for training and evaluation.

Dataset: https://www.kaggle.com/datasets/msambare/fer2013

The dataset is not included in this repository. Download it separately if you intend to retrain the model.

## Limitations

- The current model has limited predictive accuracy.
- Facial expressions do not reliably reveal a person's internal emotional state.
- Lighting, face angle, occlusion, and image quality can affect results.
- The webcam interface currently analyzes captured snapshots in the browser.

## Future Improvements

- Improve model accuracy and per-class performance.
- Evaluate predictions using a confusion matrix.
- Reduce false-positive face detections.
- Add model comparison and more robust face detection.
- Deploy the application online.

## Disclaimer

This project is intended for educational purposes. Predictions represent model classifications of visible facial expressions and should not be used to make consequential decisions about people.

## 📬 Contact

**Samiksha Raut**

📧 Email: [rautsamiksha0811@gmail.com](mailto:rautsamiksha0811@gmail.com)