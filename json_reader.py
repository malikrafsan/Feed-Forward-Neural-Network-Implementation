import json

class JsonReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None

    def read(self):
        with open(self.filename, 'r') as f:
            self.data = json.load(f)

    def get(self, index: int):
        return self.data[index]

    def length(self):
        return len(self.data)

    