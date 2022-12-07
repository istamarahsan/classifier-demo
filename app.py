from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from util.validation import FileValidator
from ML.model_frontend import *

UPLOAD_FOLDER = '/tempdir'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
file_validator = FileValidator(ALLOWED_EXTENSIONS)
model = MockupFrontend(["Orr", "hi", "computer"])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def handle_request():
    if request.method != 'POST' or 'file' not in request.files:
        return render_template('index.html')

    file = request.files['file']

    if not file or file.filename == '' or not file_validator.is_valid_filename(file.filename):
        return render_template('index.html')

    content = file.stream.read().decode("utf-8").strip()
    print(content)

    labels = model.predict_labels(content)

    return render_template("index.html", labels=labels)


if __name__ == '__main__':
    app.run()
