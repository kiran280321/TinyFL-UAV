import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------------
TFLITE_MODEL_PATH = "tinyml_model/model_quant.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 64

# --------------------------------------------------------
# PREPROCESS IMAGE (Auto-detect input shape)
# --------------------------------------------------------
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0

    model_input_shape = input_details[0]["shape"]

    # Model expects (H, W, C) → No batch dim
    if len(model_input_shape) == 3:
        return img_array.astype(np.float32)

    # Model expects (1, H, W, C) → Add batch dim
    if len(model_input_shape) == 4:
        return np.expand_dims(img_array, axis=0).astype(np.float32)

    raise ValueError(f"Unexpected model input shape: {model_input_shape}")

# --------------------------------------------------------
# TFLITE PREDICTION
# --------------------------------------------------------
def predict(image):
    img = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Handle models with different output tensor shapes
    if output.ndim == 2:
        prediction = float(output[0][0])
    else:
        prediction = float(output[0])

    label = "🚨 SMOKE DETECTED" if prediction > 0.5 else "✅ CLEAR — NO SMOKE"
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    return label, confidence

# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
st.set_page_config(
    page_title="TinyFL-UAV Smoke Detection",
    layout="centered",
    page_icon="🚁"
)

st.title("🚁 TinyFL-UAV — Edge Intelligence Inference")
st.subheader("TinyML + Federated Learning | UAV Real-Time Detection")

st.write("Upload an image to classify using your **TinyML TFLite Model**.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Run Detection"):
        label, confidence = predict(image)

        st.success(f"### {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        if "SMOKE" in label:
            st.error("⚠ UAV Alert: Possible fire/smoke detected!")
        else:
            st.info("✔ UAV Area is safe.")

st.markdown("---")
st.caption("Powered by TinyML + Federated Learning + Streamlit UI")
