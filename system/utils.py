import numpy as np
import json
import os

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    ids2query: str
    ids2response: str

def load_openai_key(key_file: str = None):
    if not key_file:
        current_path = os.path.abspath(os.getcwd())
        key_file = os.path.join(os.path.dirname(current_path), "openai_key.json") 
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            openai_key = json.load(f)
        assert isinstance(openai_key, str), f"openai_key {type(openai_key)} must be str"
        return openai_key
    else:
        raise FileNotFoundError