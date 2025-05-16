# Use python 3.9
# Use numpy 2.02 (latest supported)
# Other packages: use latest version

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "modelBannedVsGuests.tflite"
LABELS_PATH = "labelsBannedVsGuests.txt"
IMAGE_PATH = "blackDude.jpg"

#MODEL_PATH = "model.tflite"
#LABELS_PATH = "labels.txt"

# loading the labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# prep image
image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)  # Normalize

# load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor stuff
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Run stuff
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_index)[0]

#Print results
for label, confidence in zip(labels, output_data):
    print(f"{label}: {confidence:.2%}")
