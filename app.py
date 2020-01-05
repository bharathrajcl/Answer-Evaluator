import tensorflow as tf
tf.enable_eager_execution()
import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ocr_core import *


UPLOAD_FOLDER = '/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file_master' not in request.files:
            return render_template('upload.html', msg='No file selected in master file')
        file_master = request.files['file_master']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file_master.filename == '':
            return render_template('upload.html', msg='No file selected in master file')
        
        if 'file_student' not in request.files:
            return render_template('upload.html', msg='No file selected in student file')
        file_student = request.files['file_student']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file_student.filename == '':
            return render_template('upload.html', msg='No file selected in student file')

        #if file and allowed_file(file.filename):
            #file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))

            # call the OCR function on it
        extracted_text =  evaluate_student(file_master,file_student)
        print(extracted_text)

            # extract the text and display it
        return render_template('upload.html',
                               msg='Successfully processed',
                               extracted_text=extracted_text)
                               #img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True,use_reloader = False)
