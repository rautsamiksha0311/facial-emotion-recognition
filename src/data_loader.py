
from pathlib import Path
import tensorflow as tf
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "fer2013"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

def load_datasets():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Dataset folders not found at: {DATASET_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(normalize)
    val_ds = val_ds.map(normalize)
    test_ds = test_ds.map(normalize)

    print("Emotion classes:", class_names)
    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
