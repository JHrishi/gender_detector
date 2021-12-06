from flask import Flask, render_template, request, flash
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

app = Flask(__name__)
app.secret_key = "secret key"
extension = ['png', 'jpg', 'jpeg', 'gif', 'jfif']
model = load_model('gender.h5')
classes = ['M', 'F']


def gender(img_path):
    img = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face, confidence = cv.detect_face(img_rgb)
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        # crop the detected face region
        face_crop = np.copy(img[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (112,112))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]

        idx = np.argmax(conf)
        label = classes[idx]

        gen = f"{label}"
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        if gen == "M":
            c = (255,0,0)
        else:
            c = (0,255,0)
        # draw rectangle over face
        cv2.rectangle(img_rgb, (startX,startY), (endX,endY), c, 2)
        # write label and confidence above face rectangle
        cv2.putText(img_rgb, gen, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, c, 2)

    data = Image.fromarray(img_rgb)
    return data.save("static/data.jpg")


@app.route('/')
def main():
    return render_template('home.html')

@app.route("/index")
def index_page():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img.filename == '':
            flash('No image selected for uploading')
        else:
            ext = img.filename.rsplit(".",1)[1].lower()
            if ext not in extension:
                flash('Allowed image types are - png, jpg, jpeg, gif')
            else:
                img_path = "static/" + img.filename
                img.save(img_path)
                gender(img_path)
                flash(f'Image successfully detected and displayed below...')
        return render_template("index.html")


if __name__ =='__main__':
    app.run(debug = True)