import os
import cv2
import glob
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat
)

# ---- collect 250 images from all subfolders ----
IMAGE_ROOT = "data/300w/300w"
SUBFOLDERS = ["afw", "helen", "ibug", "lfpw"]
IMG_SIZE   = 112
MAX_IMAGES = 250

image_paths = []
for folder in SUBFOLDERS:
    folder_path = os.path.join(IMAGE_ROOT, folder)
    paths = glob.glob(os.path.join(folder_path, "**/*.jpg"), recursive=True)
    paths += glob.glob(os.path.join(folder_path, "**/*.png"), recursive=True)
    image_paths.extend(paths)

image_paths = image_paths[:MAX_IMAGES]
print(f"Found {len(image_paths)} calibration images")


# ---- calibration reader ----
class CalibrationReader(CalibrationDataReader):
    def __init__(self, paths):
        self.paths = paths
        self.idx   = 0

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std          # normalize
        img = img.transpose(2, 0, 1)                # HWC → CHW
        return img[np.newaxis, ...].astype(np.float32)  # add batch dim

    def get_next(self) -> dict | None: #type: ignore[override]
        if self.idx >= len(self.paths):
            return None
        arr = self.preprocess(self.paths[self.idx])
        self.idx += 1
        return {"pixel_values": arr}


reader = CalibrationReader(image_paths)

# ---- quantize ----
quantized_onnx = "models/model_int8.onnx"

quantize_static(
    model_input="models/model.onnx",
    model_output=quantized_onnx,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)

fp32_size = os.path.getsize("models/model.onnx") / 1e6
int8_size = os.path.getsize(quantized_onnx) / 1e6
print(f"FP32 : {fp32_size:.2f} MB")
print(f"INT8 : {int8_size:.2f} MB")
print(f"Reduction : {fp32_size/int8_size:.2f}×")
print("Quantized model saved to", quantized_onnx)