import cv2
import numpy as np
import inspect

def with_params(func):
    def wrapper(img, params):
        return func(img, **{k: params[k] for k in inspect.signature(func).parameters.keys() if k in params})
    return wrapper

@with_params
def contrast_stretching(img, low_clip=5.0, high_clip=97.0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    low_val, high_val = np.percentile(img, (low_clip, high_clip))
    out_img = np.uint8(np.clip((img - low_val) * 255.0 / (high_val - low_val), 0, 255))
    return out_img