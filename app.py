import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from skimage import io, feature
import cv2
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Fungsi untuk menghitung GLCM dan ekstraksi fitur
def calculate_glcm_features(image):
    # Ubah menjadi citra grayscale jika perlu
    if image.ndim > 2:
        image = image[:,:,0]

    # Hitung GLCM
    glcm = feature.greycomatrix(image, [1], [0], levels=256)

    # Ekstraksi fitur dari GLCM
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = np.zeros(len(properties))

    for i, prop in enumerate(properties):
        features[i] = feature.greycoprops(glcm, prop)

    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Periksa apakah folder 'uploads' ada
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Ambil file yang diunggah oleh pengguna
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # Simpan file ke folder 'uploads'
    file.save(filepath)

    # Baca citra
    image = io.imread(filepath)

    #resize
    image = cv2.resize(image,(224,224))

    # Hitung GLCM dan ekstraksi fitur
    features = calculate_glcm_features(image)

    #Pemodelan
    clf = joblib.load("clf.pkl")

    # Hapus file setelah selesai
    # os.remove(filepath)

    data_array = np.asarray(features)
    data_reshape = data_array.reshape(1, -1)
    prediction = clf.predict(data_reshape)
    if prediction == 1:
        result = "Iya menderita tumor otak"
    else:
        result = "Tidak menderita tumor otak "

    return render_template("index.html", result=result, filename=filepath)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
