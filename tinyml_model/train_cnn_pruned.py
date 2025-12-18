import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot

IMG_SIZE = 64
BATCH = 32
EPOCHS = 20
DATA_PATH = "../data/"

train_gen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
    class_mode="binary"
)

val_data = train_gen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
    class_mode="binary"
)

base_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

pruning_config = {
    "pruning_schedule": sparsity.PolynomialDecay(
        initial_sparsity=0.10,
        final_sparsity=0.70,
        begin_step=0,
        end_step=len(train_data) * EPOCHS
    )
}

pruned_model = sparsity.prune_low_magnitude(base_model, **pruning_config)

pruned_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    sparsity.UpdatePruningStep()
]

history = pruned_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
final_model.save("model_stripped.h5")

print("🎉 Pruned & stripped TinyML model saved as model_stripped.h5")
