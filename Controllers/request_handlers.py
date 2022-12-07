from typing import Protocol
from flask import Response, Request
from werkzeug.utils import redirect, secure_filename
from util.validation import FileValidator


class RequestHandler(Protocol):
    def handle_request(self, request: Request) -> Response:
        pass


class FileUploadHandler:
    def __init__(self, next: RequestHandler, validator: FileValidator):
        self.next = next
        self.validator = validator

    def handle_request(self, request: Request) -> Response:
        if request.method != 'POST' or 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if not file or file.filename == '' or self.validator.is_valid_filename(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        text = file.stream.read()
        print(text)
        return self.next.handle_request(request)


class EmptyResponse:
    def handle_request(self, request: Request) -> Response:
        return Response("")
