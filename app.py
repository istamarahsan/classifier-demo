from flask import Flask, request, render_template
from Middleware.classifiers import *
from Models.app_models import LabelDisplayDetails

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
    results: list[LabelModelOutput] = model.predict_labels(text)
    display: list[LabelDisplayDetails] = [LabelDisplayDetails(result) for result in results]

    return render_template("result.html", labels=display)


if __name__ == '__main__':
    app.run()
