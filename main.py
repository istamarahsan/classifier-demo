import os

import flask
from flask import Flask, request, render_template, url_for
from middleware.classifiers import *
from models.app import LabelDisplayDetails

app = Flask(__name__)
model = MLClassifier(
    ml.ml.load_model(
        model_path='static/ml/classifier_state_dict.bin',
        vocabulary_path='static/ml/vocabulary.bin',
        labelrefs_path='static/ml/labels.json'
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
