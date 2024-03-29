import numpy as np

from typing import List, Dict
from render import BaseRender

from render import YES_SINGAL, NO_SINGAL, NOT_KNOW_SINGAL, QUIT_SINGAL
from render import QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL


class UserAgent():
    def __init__(self, max_turn: int = 15, enable_not_know: bool = False, enable_quit: bool = False, 
                    num_not_know: int = 3, render: BaseRender = None):  
        self.max_turn = max_turn 
        self.enable_not_know = enable_not_know  # enable user return Not Know
        self.num_not_know = num_not_know    # if more than um_not_know items in attribute id return not know and remove attribute id
        self.enable_quit = enable_quit    # enable user quit at max_turn
        self.render = render
        
        self.turn_id = 0
        self.label_item_ids = []
        self.label_attribute_ids = {}
    
    def set_session(self, session_id: int, label_item_ids: List[int], label_attribute_ids: Dict[int, List[int]]):
        self.session_id = session_id
        self.turn_id = 0
        # label_ids: a list of all the positive item ids in session
        self.label_item_ids = label_item_ids
        self.label_attribute_ids = label_attribute_ids

        self._sort()    # for convenience to debug
    
    def _sort(self):
        self.label_item_ids.sort()
        
        self.label_attribute_ids = dict(sorted(self.label_attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.label_attribute_ids.values():
            attribute_vals.sort()

        print("LABEL ITEMS: ", self.label_item_ids)
        print("LABEL ATTRIBUTES: ", self.label_attribute_ids)
    
    def _reponse_item(self, query_item_ids: List[int]):
        label_query_item_ids = []
        for query_item_id in query_item_ids:
            if query_item_id in self.label_item_ids:
                label_query_item_ids.append(query_item_id)
        if label_query_item_ids == []:
            return NO_SINGAL
        else:
            return (YES_SINGAL, label_query_item_ids)
    
    def _reponse_attribute(self, query_attribute_id: int):
        return self.label_attribute_ids[query_attribute_id]
    
    def _reponse_attribute_with_not_know(self, query_attribute_id: int):
        if len(self.label_attribute_ids[query_attribute_id]) > self.num_not_know:
            return NOT_KNOW_SINGAL
        else:
            return self._reponse_attribute(query_attribute_id=query_attribute_id)
    
    def _reponse_attribute_value(self, query_attribute_id: int, query_attribute_vals: List[int]):
        label_attribute_values = self.label_attribute_ids[query_attribute_id]
        if set(label_attribute_values) & set(query_attribute_vals):
            return label_attribute_values
        else:
            return []

    def response(self, query_type: str, query_id):
        if self.enable_quit:
            if self.turn_id > self.max_turn:
                return QUIT_SINGAL
        self.turn_id += 1
        if query_type == QUERY_ITEM_SIGNAL:
            assert isinstance(query_id, List)
            return self._reponse_item(query_item_ids=query_id)
        elif query_type == QUERY_ATTRIBUTE_SINGAL:
            assert isinstance(query_id, int)
            if self.enable_not_know:
                return self._reponse_attribute_with_not_know(query_attribute_id=query_id)
            else:
                return self._reponse_attribute(query_attribute_id=query_id)
        elif query_type == QUERY_ATTRIBUTE_VAL_SIGNAL:
            query_attribute_id, query_attribute_vals = query_id
            return self._reponse_attribute_value(query_attribute_id=query_attribute_id, query_attribute_vals=query_attribute_vals)
        else:
            print(f"query type must be in {[QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL]}")
            raise NotImplementedError
    
    def evaluate(self, query_item_ids: List[int]):
        for query_item_id in query_item_ids:
            if query_item_id in self.label_item_ids:
                return YES_SINGAL
        return NO_SINGAL
    
    def response_with_render(self, query: str) -> str:
        assert isinstance(self.render, BaseRender)
        query_type, query_id = self.render.query2ids(query=query)
        response = self.response(query_type=query_type, query_id=query_id)
        return self.render.ids2response(response=response)