from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import json
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan nama kelas
model = load_model('model_daun.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Mapping nama penyakit
penyakit_mapping = {
    'Aloe_Anthracnose': 'Anthracnose',
    'Aloe_Healthy': 'Sehat',
    'Aloe_LeafSpot': 'Bercak Daun (Leaf Spot)',
    'Aloe_Rust': 'Karat Daun (Rust)',
    'Aloe_Sunburn': 'Terbakar Matahari (Sun Burn)',
    'Cactus_Dactylopius_Opuntia': 'Infeksi Dactylopius',
    'Cactus_Healthy': 'Sehat',
    'Money_Plant_Bacterial_wilt_disease': 'Layu Bakteri (Bacterial Wilt)',
    'Money_Plant_Healthy': 'Sehat',
    'Money_Plant_Manganese_Toxicity': 'Keracunan Mangan (Manganese Toxicity)',
    'Snake_Plant_Anthracnose': 'Anthracnose',
    'Snake_Plant_Healthy': 'Sehat',
    'Snake_Plant_Leaf_Withering': 'Layu Daun (Leaf Withering)',
    'Spider_Plant_Fungal_leaf_spot': 'Bercak Jamur (Fungal Leaf Spot)',
    'Spider_Plant_Healthy': 'Sehat',
    'Spider_Plant_Leaf_Tip_Necrosis': 'Nekrosis Ujung Daun (Leaf Tip Necrosis)',
}

def preprocessing_gambar(img_path):
    img = cv2.imread(img_path)
    # Auto brightness & contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Reduce noise
    img = cv2.GaussianBlur(img, (3,3), 0)
    cv2.imwrite(img_path, img)
    return img_path

def ekstrak_fitur_warna(img_path):
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])
    return hist_h, hist_s, hist_v

def ekstrak_fitur_morfologi(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        luas = cv2.contourArea(contour)
        keliling = cv2.arcLength(contour, True)
        return luas, keliling
    return 0, 0

def prediksi_gambar(img_path):
    # Preprocessing dulu
    img_path = preprocessing_gambar(img_path)

    # Prediksi dengan CNN
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediksi = model.predict(img_array)
    kelas_index = str(np.argmax(prediksi))
    nama_penyakit_raw = class_names.get(kelas_index, 'Tidak dikenali')
    nama_penyakit = penyakit_mapping.get(nama_penyakit_raw, nama_penyakit_raw)
    confidence = float(np.max(prediksi)) * 100

    # Ekstrak fitur warna
    hist_h, hist_s, hist_v = ekstrak_fitur_warna(img_path)
    mean_h = float(np.mean(hist_h))
    mean_s = float(np.mean(hist_s))

    # Ekstrak fitur morfologi
    luas, keliling = ekstrak_fitur_morfologi(img_path)

    return {
        'penyakit': nama_penyakit,
        'confidence': round(confidence, 2),
        'fitur_warna': {
            'mean_hue': round(mean_h, 2),
            'mean_saturation': round(mean_s, 2)
        },
        'fitur_morfologi': {
            'luas': round(luas, 2),
            'keliling': round(keliling, 2)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File kosong'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    hasil = prediksi_gambar(filepath)
    return jsonify(hasil)

if __name__ == '__main__':
    app.run(debug=True)