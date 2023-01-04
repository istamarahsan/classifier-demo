import os
import requests
from flask import Flask, request, render_template
from middleware.classifiers import *
from models.app import LabelDisplayDetails

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
STATIC_ML_DIR = os.path.join(ROOT_DIR, 'static', 'ml')
ML_STATE_PATH = os.path.join(STATIC_ML_DIR, 'classifier_state_dict.bin')
LABELREFS_PATH = os.path.join(STATIC_ML_DIR, 'labels.json')
VOCAB_PATH = os.path.join(STATIC_ML_DIR, 'vocabulary.bin')

app = Flask(__name__)

ml_state_present = os.path.isfile(ML_STATE_PATH)
labelrefs_present = os.path.isfile(LABELREFS_PATH)
vocab_present = os.path.isfile(VOCAB_PATH)
if not os.path.isdir(STATIC_ML_DIR):
    os.makedirs(STATIC_ML_DIR)

if not ml_state_present:
    url = os.getenv('ML_STATE_URL')
    response = requests.get(url)
    open(ML_STATE_PATH, "wb").write(response.content)

if not labelrefs_present:
    url = os.getenv('LABELREFS_URL')
    response = requests.get(url)
    open(LABELREFS_PATH, "wb").write(response.content)

if not vocab_present:
    url = os.getenv('VOCAB_URL')
    response = requests.get(url)
    open(VOCAB_PATH, "wb").write(response.content)

model = \
    MLClassifier(
        ml.ml.load_model(
            model_path=ML_STATE_PATH,
            vocabulary_path=VOCAB_PATH,
            labelrefs_path=LABELREFS_PATH
        )
    )


@app.get('/')
def home():
    return render_template('home.html')


@app.post('/')
def handle_request():
    if request.method != 'POST':
        return render_template('home.html')

    text = request.form.get("textf")
    results = model.predict_labels(text)
    display = [LabelDisplayDetails(result) for result in results]

    return render_template("result.html", labels=display)


if __name__ == '__main__':
    app.run(port=os.getenv("PORT", default=5000))
