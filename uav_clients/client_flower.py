import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import flwr as fl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 64
DATA_PATH = "../data/"


train_gen = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    shuffle=True
)


model = tf.keras.models.load_model("../tinyml_model/model_stripped.h5")
print("✅ Loaded model_stripped.h5 successfully.")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print("🔧 Model compiled and ready for training.")


class UAVClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("🚁 UAV Client: Starting local training...")

        model.set_weights(parameters)
        model.fit(train_data, epochs=1, verbose=1)

        print("📡 Uploading weights back to server (no compression sent)...")
        return model.get_weights(), len(train_data), {}

    def evaluate(self, parameters, config):
        print("📊 UAV Client: Evaluating global model...")

        model.set_weights(parameters)
        loss, acc = model.evaluate(train_data, verbose=0)

        print(f"🔍 Evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")
        return loss, len(train_data), {"accuracy": acc}


print("🚀 UAV Client connecting to Federated Server at 127.0.0.1:8080")

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=UAVClient().to_client()
)
