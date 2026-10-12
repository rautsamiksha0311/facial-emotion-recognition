
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if FACE_CASCADE.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")

    model = tf.keras.models.load_model(MODEL_PATH)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print("No face detected. Try a clearer image with good lighting.")
        return

    for face_number, (x, y, w, h) in enumerate(faces, start=1):
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32) / 255.0
        face = face.reshape(1, 48, 48, 1)

        probabilities = model.predict(face, verbose=0)[0]

        if len(probabilities) != len(EMOTION_LABELS):
            raise ValueError(
                f"Expected 7 model outputs, received {len(probabilities)}."
            )

        emotion_index = int(np.argmax(probabilities))
        emotion = EMOTION_LABELS[emotion_index]
        confidence = float(probabilities[emotion_index]) * 100

        print(f"\nFace {face_number} — all emotion scores:")
        for label, score in zip(EMOTION_LABELS, probabilities):
            print(f"{label.title():10s}: {score * 100:.2f}%")

        print(f"Predicted expression: {emotion.title()}")
        print(f"Highest model score: {confidence:.2f}%")

        cv2.rectangle(
            image, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            f"{emotion.title()}: {confidence:.1f}%",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Facial Emotion Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip().strip('"')
    predict_image(image_path)
