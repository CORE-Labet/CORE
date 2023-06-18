import openai

from typing import Dict, List

OPENAI_API_KEY = ""

def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


class BaseRender():

    def encode(text: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError

    def decode(query_type: str, query_id) -> str:
        raise NotImplementedError


class LMRender(BaseRender):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name    
        
    def encode(text: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError

    def decode(query_type: str, query_id) -> str:
        raise NotImplementedError