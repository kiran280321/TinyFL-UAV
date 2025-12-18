import tensorflow as tf

model = tf.keras.models.load_model("model_stripped.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model_micro.tflite", "wb") as f:
    f.write(tflite_model)

def convert_to_c_array(tflite_model):
    hex_array = ', '.join(['0x{:02x}'.format(b) for b in tflite_model])
    c_code = f"""\
#include <stdint.h>
const unsigned char model_micro[] = {{
{hex_array}
}};
const unsigned int model_micro_len = {len(tflite_model)};
"""
    return c_code

c_code = convert_to_c_array(tflite_model)

with open("model_micro.cc", "w") as f:
    f.write(c_code)

print("🎉 Saved TinyML models:")
print(" - model_micro.tflite")
print(" - model_micro.cc (for microcontrollers)")
