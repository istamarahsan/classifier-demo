import os

from flask import Flask, request, render_template
from middleware.classifiers import *
from models.app import LabelDisplayDetails

app = Flask(__name__)
model = RandomClassifier()


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def handle_request():
    if request.method != 'POST':
        return render_template('home.html')

    text: str = request.form.get("textf")
    results = model.predict_labels(text)
    display = [LabelDisplayDetails(result) for result in results]

    return render_template("result.html", labels=display)


if __name__ == '__main__':
    app.run(port=os.getenv("PORT", default=5000))
