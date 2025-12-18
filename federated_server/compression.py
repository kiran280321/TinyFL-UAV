import numpy as np

def quantize_weights(w, bits=8):
    scale = np.max(np.abs(w))
    if scale == 0:
        scale = 1e-6
    q = np.round((w / scale) * (2**(bits-1)-1))
    return q.astype(np.int8), scale


def dequantize_weights(q, scale):
    return (q.astype(np.float32) / (2**7)) * scale
