import openai

from dataclasses import dataclass
from typing import Dict, List
from utils import load_openai_key



@dataclass
class PromptTemplate:
    ids2querypre: str
    id2querypost: str
    ids2responsepre: str
    ids2responsepost: str


def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


class BaseRender():
    def __init__(self, template: PromptTemplate = None):
        self.template = template
    
    def set_session(self, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        self.item_ids = item_ids
        self.attribute_ids = attribute_ids
    
    def response2ids(self, response: str):
        raise NotImplementedError
    
    def ids2response(self, response_type: str, response_id) -> str:
        raise NotImplementedError

    def ids2query(self, query_type: str, query_id) -> str:
        raise NotImplementedError
    
    def query2ids(self, query: str):
        raise NotImplementedError


class RuleRender(BaseRender):
    def __init__(self, template: PromptTemplate = None):
        super().__init__(template=template)
        if not template:
            self.template = PromptTemplate
        
    def response2ids(self, response: str):
        res = []
        start_index = response.find(self.template.ids2responsepre) + 1
        end_index = response.find(self.template.ids2responsepost)
        text = response[start_index:end_index].split(",")
        for idx in text:
            if int(idx) in self.item_ids:
                res.append(idx)
        return res
    
    def ids2response(self, response_type: str, response_ids) -> str:
        text = ""
        for response_id in response_ids:
            text += str(response_id)
        return response_type + self.template.ids2responsepre + text + self.template.ids2responsepost

    def ids2query(self, query_type: str, query_ids) -> str:
        text = ""
        for query_id in query_ids:
            text += str(query_id)
        return query_type + self.template.ids2querypre + text + self.template.id2querypost
    
    def query2ids(self, query: str):
        res = []
        start_index = query.find(self.template.ids2querypre) + 1
        end_index = query.find(self.template.id2querypost)
        text = query[start_index:end_index].split(",")
        for idx in text:
            if int(idx) in self.item_ids:
                res.append(idx)
        return res
    

class LMRender(BaseRender):
    def __init__(self, template: PromptTemplate = None, model_name: str = "gpt-3.5-turbo", key_file: str = None):
        super().__init__(template=template)
        self.model_name = model_name
        openai.api_key = load_openai_key(key_file=key_file) 
        if not template:
            self.template = PromptTemplate()   
        
    def response2ids(self, response: str):
        raise NotImplementedError
    
    def ids2response(self, response_type: str, response_id) -> str:
        raise NotImplementedError

    def ids2query(self, query_type: str, query_id) -> str:
        raise NotImplementedError
    
    def query2ids(self, query: str):
        raise NotImplementedError


if __name__ == "__main__":
    template = PromptTemplate
