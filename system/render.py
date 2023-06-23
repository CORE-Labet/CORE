import openai

from dataclasses import dataclass
from typing import Dict, List, Optional
from utils import load_openai_key
from agent import QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL
from user import YES_SINGAL, NO_SINGAL, QUIT_SINGAL


@dataclass
class PromptTemplate:
    ids2querypre: Optional[str] = None
    id2querypost: Optional[str] = None
    ids2response: Optional[str] = None


def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


class BaseRender():
    def __init__(self, templates: Dict = None):
        self.templates = templates
    
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
    def __init__(self, templates: PromptTemplate = None):
        super().__init__(templates=templates)
        if not templates:
            self.templates = {}
            self.templates[QUERY_ITEM_SIGNAL] = PromptTemplate(
                ids2querypre="what items could you prefer, we would like recommend: ",
                ids2querypost=f"; please click ({YES_SINGAL}) or select to re-generate ({NO_SINGAL})."
            )
            self.templates[QUERY_ATTRIBUTE_SINGAL] = PromptTemplate(
                ids2querypre="may we know your preferences on following attribute: ",
                ids2querypost="; please response with specific values to help seeking recommendations."
            )
            self.templates[QUERY_ATTRIBUTE_VAL_SIGNAL] = PromptTemplate(
                ids2querypre="may we know your preferences on following attribute values: ",
                ids2querypost="; please select your prefered attribute values or response with an empty set if there is no preferred value."
            )
            self.templates[YES_SINGAL] = PromptTemplate(
                ids2response="thanks, I have found my preferred items.",
            )
            self.templates[NO_SINGAL] = PromptTemplate(
                ids2response="sorry, there is no item satisfying my needs.",
            )
            self.templates[QUIT_SINGAL] = PromptTemplate(
                ids2response="too many turns, I want to quit."
            )
        
    def response2ids(self, response: str):
        res = []
        if response.startswith(QUIT_SINGAL):
            return QUIT_SINGAL
        if response.startswith(YES_SINGAL):
            start_index = response.find(self.templates[YES_SINGAL]) + 1
        elif response.startswith(NO_SINGAL):
            start_index = response.find(self.templates[YES_SINGAL]) + 1
        else:
            print(f"response {response} must start with either {YES_SINGAL}, {NO_SINGAL}, or {QUIT_SINGAL}")
            raise ValueError
        text = response[start_index:].split(",")
        for idx in text:
            if int(idx) in self.item_ids:
                res.append(idx)
        return res
    
    def ids2response(self, response_type: str, response_ids) -> str:
        assert response_type in [YES_SINGAL, NO_SINGAL, QUIT_SINGAL], f"response_type {response_type} must be either {YES_SINGAL}, {NO_SINGAL}, or {QUIT_SINGAL}."
        text = ""
        for response_id in response_ids:
            text += str(response_id)
        return response_type + self.templates[response_type].ids2response + text

    def ids2query(self, query_type: str, query_ids) -> str:
        assert query_type in [QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL], f"query_type {query_type} must be either {QUERY_ITEM_SIGNAL}, {QUERY_ATTRIBUTE_SINGAL}, or {QUERY_ATTRIBUTE_VAL_SIGNAL}."
        text = ""
        for query_id in query_ids:
            text += str(query_id)
        return query_type + self.templates[query_type].ids2querypre + text + self.templates[query_type].ids2querypost
    
    def query2ids(self, query: str):
        res = []
        if query.startswith(QUERY_ITEM_SIGNAL):
            start_index = query.find(self.templates[QUERY_ITEM_SIGNAL].ids2querypre) + 1
            end_index = query.find(self.templates[QUERY_ITEM_SIGNAL].id2querypost)
        elif query.startswith(QUERY_ATTRIBUTE_SINGAL):
            start_index = query.find(self.templates[QUERY_ATTRIBUTE_SINGAL].ids2querypre) + 1
            end_index = query.find(self.templates[QUERY_ATTRIBUTE_SINGAL].id2querypost)
        elif query.startswith(QUERY_ATTRIBUTE_VAL_SIGNAL):
            start_index = query.find(self.templates[QUERY_ATTRIBUTE_VAL_SIGNAL].ids2querypre) + 1
            end_index = query.find(self.templates[QUERY_ATTRIBUTE_VAL_SIGNAL].id2querypost)
        else:
            print(f"query {query} must start with either {QUERY_ITEM_SIGNAL}, {QUERY_ATTRIBUTE_SINGAL}, or {QUERY_ATTRIBUTE_VAL_SIGNAL}.")
            raise ValueError
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
    render = RuleRender()
