import numpy as np

from typing import List, Dict
from collections import Counter
from checker import BaseChecker
from trainer import BaseTrainer
from render import BaseRender

QUERY_ITEM_SIGNAL = "item"
QUERY_ATTRIBUTE_SINGAL = "attribute"
QUERY_ATTRIBUTE_VAL_SIGNAL = "attribute_val"

class ConversationalAgent():
    def __init__(self, checker: BaseChecker = None, trainer: BaseTrainer = None,
                    render: BaseRender = None, cold_start: bool = False):
        self.checker = checker
        self.trainer = trainer
        self.render = render

        self.num_turn = 0
        self.cold_start = cold_start
        self.data_matrix = None
        self.checked_item_ids = []  # record checked items
        self.item_ids = []
        self.attribute_ids = {}

    # go to next sesssion
    def set_session(self, session_id: int, data_matrix: np.ndarray):
        self.num_turn = 0
        self.session_id = session_id
        self.data_matrix = data_matrix
        if self.cold_start: # if cold-start, there is no labels
            self.data_matrix[:, -1] = 1

        self.item_ids = [_ for _ in range(self.data_matrix.shape[0])]
        self.attribute_ids = {} # reset
        attribute_ids = [_ for _ in range(self.data_matrix.shape[1] - 1)]  # last col is label
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            self.attribute_ids.update({attribute_id: list(attribute_dict.keys())})
        self.checked_item_ids = list()  # reset
    
    # go to next turn
    def set_turn(self, item_ids: List[int], attribute_ids: Dict[int, List[int]]):
        checked_item_ids = list(set(self.item_ids) - set(item_ids))
        self.checked_item_ids.extend(checked_item_ids)
        
        self.item_ids = item_ids
        self.attribute_ids = attribute_ids
        self.num_turn += 1

        self.checked_item_ids.sort()
        self.item_ids.sort()
        self.attribute_ids = dict(sorted(self.attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.attribute_ids.values():
            attribute_vals.sort()

    def check(self):
        query_type, query_id = self.checker.act(data_matrix=self.data_matrix, item_ids=self.item_ids, attribute_ids=self.attribute_ids, num_turn=self.num_turn)  
        return (self.item_ids, self.attribute_ids, query_type, query_id)
    
    def evaluate(self):
        query_type, query_id = self.checker.evaluate(data_matrix=self.data_matrix, item_ids=self.item_ids)
        assert query_type == QUERY_ITEM_SIGNAL, f"during evaluation, query type {query_type} must be {QUERY_ITEM_SIGNAL}"
        return query_id

    def load(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def check_with_render(self, response: str) -> str:
        assert isinstance(self.render, BaseRender)
        item_ids, attribute_ids = self.render.response2ids(response=response)
        self.set_turn(item_ids=item_ids, attribute_ids=attribute_ids)
        _, _, query_type, query_id = self.check()
        raise self.render.ids2query(query_type=query_type, query_id=query_id)