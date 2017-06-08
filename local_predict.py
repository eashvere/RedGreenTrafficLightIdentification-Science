import os
import sys
import io
import logging
import numpy as np
import operator
import keras
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import cv2

desired_width = 160
desired_height = 160

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG'])

labels = None
project = 'trafficlightdeeplearning'
model_name = 'traffic_light'
model = keras.models.load_model('traffic-light_keras.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print "upload file"
    if request.method == 'POST':

        file = request.files['file']
        print file
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'],
                      filename))
            fname = "%s/%s" % (UPLOAD_FOLDER, filename)

            result = model.predict(load_image(fname))
            #show_json(result)
            print('\n')
            print(result)
            print('\n')

            for i in predictions:
                label, score = max(enumerate(i), key=operator.itemgetter(1))
            label = labels[label]
            print(label, score)
            return redirect(url_for('show_result',
                                    filename=filename,
                                    label=label,
                                    score=score))

    return render_template('index.html')


@app.route('/result')
def show_result():
    print "uploaded file"

    filename = request.args['filename']
    label = request.args['label']
    score = request.args['score']

    # This result handling logic is hardwired for the "hugs/not-hugs"
    # example, but would be easy to modify for some other set of
    # classification labels.
    if label == 'red':
        return render_template('jresults.html',
                               filename=filename,
                               label="RED",
                               score=score,
                               border_color="#B20000")

    elif label == 'green':
        return render_template('jresults.html',
                               filename=filename,
                               label="GREEN.",
                               score=score,
                               border_color="#00FF48")
    else:
        return render_template('error.html',
                               message="Something went wrong.")


def read_dictionary(path):
  with open(path) as f:
    return f.read().splitlines()

#dicts = 'static/dict.txt'

def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (desired_width, desired_height))
    img = np.reshape(img, (-1, desired_width, desired_height, 3))
    return img

#labels = read_dictionary(dicts)
labels = ['red', 'green']
print("labels: %s" % labels)
    # Runs on port 5000 by default.
    # Change to app.run(host='0.0.0.0') for an externally visible
    # server.
'''app.secret_key = 'super_secret_key'
app.debug = True'''
if __name__ == "__main__":
    print("labels: %s" % labels)
    app.run(host='0.0.0.0', port=4000, debug=True)
