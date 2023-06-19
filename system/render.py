import openai

from typing import Dict, List
from utils import Template

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
    
    def response2ids(self, response: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError
    
    def ids2response(self, query_type: str, query_id, response_ids) -> str:
        raise NotImplementedError

    def ids2query(self, query_type: str, query_id) -> str:
        raise NotImplementedError
    
    def query2ids(self, query: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError


class RuleRender(BaseRender):
    def __init__(self, template: Template):
        self.model_name = template    
        
    def response2ids(self, response: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError
    
    def ids2response(self, query_type: str, query_id, response_ids) -> str:
        raise NotImplementedError

    def ids2query(self, query_type: str, query_id) -> str:
        raise NotImplementedError
    
    def query2ids(self, query: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError
    

class LMRender(BaseRender):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name    
        
    def response2ids(self, response: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError
    
    def ids2response(self, query_type: str, query_id, response_ids) -> str:
        raise NotImplementedError

    def ids2query(self, query_type: str, query_id) -> str:
        raise NotImplementedError
    
    def query2ids(self, query: str, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        raise NotImplementedError