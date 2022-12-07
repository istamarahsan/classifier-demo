class FileValidator:
    def __init__(self, allowed_extensions: set):
        self.allowed_extensions = allowed_extensions

    def is_valid_filename(self, file_name: str):
        return '.' in file_name and \
            file_name.rsplit('.', 1)[1].lower() in self.allowed_extensions
