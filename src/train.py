
from pathlib import Path
import tensorflow as tf
from src.data_loader import load_datasets
from src.model import build_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks
    )

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(str(MODEL_DIR / "final_model.keras"))

if __name__ == "__main__":
    main()
