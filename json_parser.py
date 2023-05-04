from json_reader import JsonReader
from model_types import *

class JsonParser:
    def parse_model_config(self, json_path: str) -> ModelConfig:
        model_config: ModelConfig = JsonReader(json_path).read()
        return model_config
        
        