import numpy as np

from collections import Counter
from checker import BaseChecker


class ConversationalAgent():
    def __init__(self, checker_name, cold_start: bool = False, query_attribute_val: bool = False):
        self.checker = BaseChecker(checker_name=checker_name, query_attribute_val=query_attribute_val)
        self.num_turn = 0
        self.cold_start = cold_start
        self.query_attribute_val = query_attribute_val # query attribute value instead of attribute id
        # initial
        self.data_matrix, self.item_ids, self.attribute_ids = None, None, None
        # record checked items
        self.checked_item_ids = []

    # go to next sesssion
    def set_session(self, session_id, data_matrix):
        self.num_turn = 0
        self.session_id = session_id
        self.data_matrix = data_matrix
        # item_ids and attribute_ids are list of ids
        self.item_ids = [_ for _ in range(self.data_matrix.shape[0])]
        self.attribute_ids = [_ for _ in range(self.data_matrix.shape[1])]
        # if cold-start, there is no labels
        if self.cold_start:
            self.data_matrix[:, -1] = 1
        if self.value_querier:
            self.attribute_vals = []
            for attribute_id in self.attribute_ids:
                attribute_dict = Counter(data_matrix[:,attribute_id])
                self.attribute_vals.append(list(attribute_dict.keys()))
        # re-initial
        self.checked_item_ids = list()
    
    # go to next turn
    def set_turn(self, item_ids, attribute_ids):
        if self.value_querier:
            self.attribute_ids, self.attribute_vals = attribute_ids
        else:
            self.attribute_ids = attribute_ids
        checked_item_ids = list(set(self.item_ids) - set(item_ids))
        self.checked_item_ids.extend(checked_item_ids)
        self.checked_item_ids.sort() # convenience for debug
        self.item_ids = item_ids
        self.num_turn += 1

    def act_checker(self):
        if self.value_querier:
            attribute_ids = (self.attribute_ids, self.attribute_vals)
        else:
            attribute_ids = self.attribute_ids
        query_type, query_id = self.checker.act(self.data_matrix, self.item_ids, attribute_ids, self.num_turn)  
        return (self.item_ids, attribute_ids, query_type, query_id)
    
    def act_evaluator(self):
        query_id = self.checker.evaluate(self.data_matrix, self.item_ids, self.attribute_ids)
        return query_id