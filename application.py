# application.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, pickle
import numpy as np
import cv2

# ใช้ Keras แบบเดียวกับตอนเทรน
from keras import mixed_precision
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                          Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# CONFIG
# -------------------------
application = Flask(__name__)
CORS(
    application,
    resources={r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://nichapintong.github.io"
        ]
    }},
    supports_credentials=False,
    allow_headers=["Content-Type"],
    methods=["GET","POST","OPTIONS"]
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

WEIGHTS_PATH = "alexnet_compact_fp16.weights.h5"  # << weights-only ตามที่ต้องการ
LABEL_TRANSFORM_PATH = "label_transform.pkl"
IMAGE_SIZE = (227, 227)

# class meta (ถ้ามีคลาสที่ไม่อยู่ใน mapping จะเติมค่าเริ่มต้นให้)
class_mapping = {
    "Brown_Spot": {"ID": 1, "nameEN": "Brown Spot", "nameTH": "โรคใบจุดสีน้ำตาล"},
    "Common_Rust": {"ID": 2, "nameEN": "Common Rust", "nameTH": "โรคราสนิม"},
    "Downy_Mildew": {"ID": 3, "nameEN": "Downy Mildew", "nameTH": "โรคราน้ำค้าง"},
    "Leaf_Spot": {"ID": 4, "nameEN": "Leaf Spot", "nameTH": "โรคใบจุด"},
    "Small_Leaf_Blight": {"ID": 5, "nameEN": "Small Leaf Blight", "nameTH": "โรคใบไหม้แผลเล็ก"},
    "Large_Leaf_Blight": {"ID": 6, "nameEN": "Large Leaf Blight", "nameTH": "โรคใบไหม้แผลใหญ่"},
    "SCMV_MDMV": {"ID": 7, "nameEN": "SCMV & MDMV", "nameTH": "โรคSCMV&MDMV"},
    "Healthy": {"ID": 8, "nameEN": "Healthy", "nameTH": "ไม่เป็นโรค"}
}

# -------------------------
# Mixed precision ให้ตรงกับตอนเทรน
# -------------------------
mixed_precision.set_global_policy('mixed_float16')  # FP16 ภายในชั้น, เอาต์พุตชั้นสุดท้ายเป็น float32

# -------------------------
# LOAD LABELS ก่อน เพื่อรู้จำนวนคลาสที่แท้จริง
# -------------------------
if not os.path.exists(LABEL_TRANSFORM_PATH):
    raise FileNotFoundError(f"Label binarizer not found: {LABEL_TRANSFORM_PATH}")

with open(LABEL_TRANSFORM_PATH, "rb") as f:
    label_binarizer = pickle.load(f)

CLASSES = list(getattr(label_binarizer, "classes_", []))
NUM_CLASSES = len(CLASSES)
if NUM_CLASSES == 0:
    raise RuntimeError("No classes found in label_binarizer.")

# เติม mapping ให้คลาสที่อาจตกหล่น
for cls in CLASSES:
    class_mapping.setdefault(cls, {"ID": -1, "nameEN": cls, "nameTH": cls})

# -------------------------
# REBUILD MODEL ให้ "เหมือนตอนเทรน"
# -------------------------
def create_compact_alexnet(num_classes):
    # ตรงกับโค้ดเทรน: Conv1=64, Conv2=192, ต่อด้วย 256,256,192, GAP, Dense(256), Dropout(0.5), Dense(num_classes, dtype=float32)
    model = Sequential([
        # Block 1
        Conv2D(64, (11, 11), strides=(4, 4), activation="relu",
               padding="valid", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((3, 3), strides=(2, 2)),
        BatchNormalization(),

        # Block 2
        Conv2D(192, (5, 5), activation="relu", padding="same"),
        MaxPooling2D((3, 3), strides=(2, 2)),
        BatchNormalization(),

        # Block 3-5
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        Conv2D(192, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((3, 3), strides=(2, 2)),

        # Head
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),

        # เอาต์พุตบังคับ dtype เป็น float32 (เหมือนตอนเทรน)
        Dense(num_classes, activation="softmax", dtype='float32')
    ])
    return model

model = create_compact_alexnet(NUM_CLASSES)

# -------------------------
# LOAD WEIGHTS (strict)
# -------------------------
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

# โหลดแบบ strict—ถ้าโค้ดตรงตามที่คุณส่งมา จะโหลดผ่าน
model.load_weights(WEIGHTS_PATH)
print("[INFO] Weights loaded successfully.")

# -------------------------
# INFERENCE (พรีโปรเซสให้เหมือนตอนเทรน)
# หมายเหตุ: ตอนเทรนใช้ cv2.imread + resize + img_to_array โดย "ไม่แปลงเป็น RGB"
# ดังนั้นคง BGR ไว้เหมือนเดิม และ normalize/astype เป็น float16
# -------------------------
def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Cannot read the uploaded image.")

    # *** ไม่แปลง BGR->RGB เพื่อให้ตรงกับตอนเทรน ***
    image = cv2.resize(image, IMAGE_SIZE)
    arr = img_to_array(image).astype(np.float16) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]  # float32 จากชั้นสุดท้าย
    order = np.argsort(preds)[::-1]

    results = []
    for i in order:
        cls_name = CLASSES[i]
        meta = class_mapping.get(cls_name, {"ID": -1, "nameEN": cls_name, "nameTH": cls_name})
        results.append({
            "ID": meta["ID"],
            "nameEN": meta["nameEN"],
            "nameTH": meta["nameTH"],
            "confidence": round(float(preds[i]) * 100.0, 2),
        })
    return results

# -------------------------
# ROUTES
# -------------------------
@application.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(application.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    try:
        results = classify_image(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    return jsonify(results)

@application.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "num_classes": NUM_CLASSES, "classes": CLASSES})

# -------------------------
# MAIN
# -------------------------
if __name__ == '__main__':
    # ถ้ารำคาญ log oneDNN: set TF_ENABLE_ONEDNN_OPTS=0 ก่อนรัน
    application.run(
        host="0.0.0.0",  # listen on all network interfaces (so EC2 clients can reach it)
        port=5000,       # expose on port 5000
        debug=True,      # enables debugger + auto error pages
        use_reloader=False  # avoid double-start issues when debug=True
    )