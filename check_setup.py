
import cv2
import tensorflow as tf
import numpy as np

print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Setup completed successfully!")
