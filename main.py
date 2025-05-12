import numpy as np
from PIL import Image
import tensorflow as tf

# --- Load your model and labels ---
MODEL_PATH = "modelBannedVsGuests.tflite"
LABELS_PATH = "labelsBannedVsGuests.txt"
IMAGE_PATH = "IMG20250505123132.jpg"

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load and prepare image
image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)  # Normalize

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Run inference
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_index)[0]

# Print results
for label, confidence in zip(labels, output_data):
    print(f"{label}: {confidence:.2%}")
