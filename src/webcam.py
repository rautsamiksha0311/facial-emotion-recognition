
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                face = face.astype(np.float32) / 255.0
                face = face.reshape(1, 48, 48, 1)

                scores = model.predict(face, verbose=0)[0]
                index = int(np.argmax(scores))
                label = f"{EMOTIONS[index]}: {scores[index] * 100:.1f}%"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x, max(y-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            cv2.imshow("Live Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
