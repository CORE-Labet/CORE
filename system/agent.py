import numpy as np

from typing import List, Dict
from collections import Counter
from checker import BaseChecker
from trainer import BaseTrainer
from render import BaseRender
from user import YES_SINGAL, NO_SINGAL, NOT_KNOW_SINGAL

QUERY_ITEM_SIGNAL = "item"
QUERY_ATTRIBUTE_SINGAL = "attribute"
QUERY_ATTRIBUTE_VAL_SIGNAL = "attribute_val"


class ConversationalAgent():
    def __init__(self, checker: BaseChecker, trainer: BaseTrainer, render: BaseRender, cold_start: bool = True):
        self.checker = checker
        self.trainer = trainer
        self.render = render

        self.turn_id = 0
        self.cold_start = cold_start
        self.data_matrix = None
        self.item_ids = []
        self.attribute_ids = {}

        self.checked_item_ids = []  # record checked items
        self.checked_attribute_ids = {} # record checked attributes

    def set_session(self, session_id: int, data_matrix: np.ndarray):
        self.turn_id = 0
        self.session_id = session_id
        if self.cold_start:
            predictions = np.zeros(data_matrix.shape[1])
        else:
            predictions = self.trainer()
        self.data_matrix = np.stack((data_matrix, predictions), axis=1)

        self.item_ids = data_matrix[:,0]
        self.attribute_ids = {} 
        attribute_ids = [_ for _ in range(1, self.data_matrix.shape[1] - 1)]  # 1st is item_id, last col is label
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            self.attribute_ids.update({attribute_id: list(attribute_dict.keys())})
        
        self.checked_item_ids = [] 
        self.checked_attribute_ids = {}
    
    def set_turn(self, query_type: str, query_id, response):
        self.turn_id += 1
        
        if query_type is QUERY_ITEM_SIGNAL:
            assert isinstance(response, List)
            self._update_item(query_item_ids=query_id)
        elif query_type is QUERY_ATTRIBUTE_SINGAL:
            if response is NOT_KNOW_SINGAL:
                self.checked_attribute_ids.update({query_id: self.attribute_ids[query_id]})
                self.attribute_ids.pop(query_id)
            else:
                assert isinstance(response, List)
                self._update_attribute(query_attribute_id=query_id, response_vals=response)
        elif query_type is QUERY_ATTRIBUTE_VAL_SIGNAL:
            assert isinstance(response, List)
            query_attribute_id, query_attribute_vals = query_id
            self._update_attribute_val(query_attribute_id=query_attribute_id, query_attribute_vals=query_attribute_vals, response_vals=response)
        else:
            print(f"query type must be in {[QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL]}")
            raise NotImplementedError
        
        self._sort()    # for convenience to debug

    def _sort(self):
        self.checked_item_ids.sort()
        self.item_ids.sort()
        
        self.checked_attribute_ids = dict(sorted(self.checked_attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.checked_attribute_ids.values():
            attribute_vals.sort()
        self.attribute_ids = dict(sorted(self.attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.attribute_ids.values():
            attribute_vals.sort()
        
    def _update_item(self, query_item_ids: List[int]):
        for query_item_id in query_item_ids:
            self.item_ids.remove(query_item_id)
    
    def _update_attribute(self, query_attribute_id: int, response_vals: List[int]):
        self.checked_attribute_ids.update({query_attribute_id: self.attribute_ids[query_attribute_id]})
        self.attribute_ids.pop(query_attribute_id) 
 
        label_data_matrix = self.data_matrix[self.data_matrix[:,query_attribute_id] in response_vals]
        label_item_ids = label_data_matrix[:,0].tolist()
        self.checked_item_ids.extend([idx for idx in self.item_ids if idx not in label_item_ids])
        self.item_ids = list(set(self.item_ids) & set(label_item_ids))
    
    def _update_attribute_val(self, query_attribute_id: int, query_attribute_vals: List[int], response_vals: List[int]):
        self.checked_attribute_ids.update({query_attribute_id: query_attribute_vals})
        if self.attribute_ids[query_attribute_id] == query_attribute_vals:
            self.attribute_ids.pop(query_attribute_id)
        else:
            self.attribute_ids[query_attribute_id] = [idx for idx in self.attribute_ids[query_attribute_id] if idx not in query_attribute_vals]
        
        label_data_matrix = self.data_matrix[self.data_matrix[:,query_attribute_id] in response_vals]
        label_item_ids = label_data_matrix[:,0].tolist()
        self.checked_item_ids.extend([idx for idx in self.item_ids if idx not in label_item_ids])
        self.item_ids = list(set(self.item_ids) & set(label_item_ids))

    def check(self):
        return self.checker.act(data_matrix=self.data_matrix, item_ids=self.item_ids, attribute_ids=self.attribute_ids, turn_id=self.turn_id)  
    
    def evaluate(self):
        query_type, query_id = self.checker.evaluate(data_matrix=self.data_matrix, item_ids=self.item_ids)
        assert query_type is QUERY_ITEM_SIGNAL, f"during evaluation, query type {query_type} must be {QUERY_ITEM_SIGNAL}"
        return query_id
    
    def train(self):
        raise NotImplementedError  
    
    def check_with_render(self, response: str) -> str:
        assert isinstance(self.render, BaseRender)
        item_ids, attribute_ids = self.render.response2ids(response=response)
        self.set_turn(item_ids=item_ids, attribute_ids=attribute_ids)
        _, _, query_type, query_id = self.check()
        raise self.render.ids2query(query_type=query_type, query_id=query_id)