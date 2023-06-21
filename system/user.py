import numpy as np

from typing import List, Dict
from agent import QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL
from render import BaseRender

SUCCESS_SIGNAL = "success"
FAILURE_SIGNAL = "failure"
QUIT_SINGAL = "quit"
NOT_KNOW_SINGAL = "not_know"


class UserAgent():
    def __init__(self, max_turn: int = 15, enable_not_know: bool = False, enable_quit: bool = False, num_not_know: int = 3, render: BaseRender = None):  
        self.max_turn = max_turn 
        self.enable_not_know = enable_not_know  # enable user return Not Know
        self.num_not_know = num_not_know    # if more than um_not_know items in attribute id return not know and remove attribute id
        self.enable_quit = enable_quit    # enable user quit at max_turn
        self.render = render

        self.label_ids, self.data_matrix, self.label_data_matrix = None, None, None
    
    def set_session(self, session_id: int, data_matrix: np.ndarray, label_ids: List[int]):
        self.session_id = session_id
        # label_ids: a list of all the positive item ids in session
        self.label_ids = label_ids
        self.data_matrix = data_matrix
        self.label_data_matrix = data_matrix[label_ids,:]
    
    def _reponse_item(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query_item_ids: List[int]):
        for query_item_id in query_item_ids:
            item_ids.remove(query_item_id)
            if query_item_id in self.label_ids:
                return SUCCESS_SIGNAL
            else:
                return (item_ids, attribute_ids)
    
    def _reponse_attribute(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query_attribute_id: int):
        label_attribute_values = list(self.label_data_matrix[:,query_attribute_id])
        label_mask = np.isin(self.data_matrix[:,query_attribute_id], label_attribute_values) # select items equal to labels
        item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
        label_item_ids = item_ids_matrix[label_mask]
        item_ids = list(set(item_ids) & set(label_item_ids))
        attribute_ids.pop(query_attribute_id) # remove queried attributes
        return (item_ids, attribute_ids)
    
    def _reponse_attribute_with_not_know(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query_attribute_id: int):
        label_attribute_values = list(set(self.label_data_matrix[:,query_attribute_id]))
        if len(label_attribute_values) > self.num_not_know:
            attribute_ids.pop(query_attribute_id)
            return (item_ids, attribute_ids)
        else:
            return self._reponse_attribute(item_ids=item_ids, attribute_ids=attribute_ids, query_attribute_id=query_attribute_id)
    
    def _reponse_attribute_value(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query_attribute_id: int, query_attribute_vals: List[int]):
        label_attribute_values = list(self.label_data_matrix[:,query_attribute_id])
        query_label_attribute_values = list(set(query_attribute_vals) & set(label_attribute_values))
        if query_label_attribute_values:
            label_mask = self.data_matrix[:,query_attribute_id] in query_label_attribute_values
            item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
            label_item_ids = item_ids_matrix[label_mask]
            item_ids = list(set(item_ids) & set(label_item_ids))
        else:
            not_label_mask = self.data_matrix[:,query_attribute_id] in query_attribute_vals
            item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
            not_label_item_ids = item_ids_matrix[not_label_mask]
            item_ids = list(set(item_ids) - set(not_label_item_ids))
        
        attribute_ids[query_attribute_id] = [attribute_val for attribute_val in attribute_ids[query_attribute_id] if attribute_val not in query_attribute_vals]
        if not attribute_ids[query_attribute_id]:
            attribute_ids.pop(query_attribute_id) # if all the values are queried, remove attribute
        
        # only store label_ids in item_ids
        self.label_ids = list(set(self.label_ids) & set(item_ids))
        self.label_data_matrix = self.data_matrix[self.label_ids,:]
        return (item_ids, attribute_ids)

    def response(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query_type: str, query_id, num_turn: int):
        if self.enable_quit:
            if num_turn > self.max_turn:
                return QUIT_SINGAL
        if query_type is QUERY_ITEM_SIGNAL:
            assert isinstance(query_id, List)
            return self._reponse_item(item_ids=item_ids, attribute_ids=attribute_ids, query_id=query_id)
        elif query_type is QUERY_ATTRIBUTE_SINGAL:
            assert isinstance(query_id, int)
            if self.enable_not_know:
                return self._reponse_attribute_with_not_know(item_ids=item_ids, attribute_ids=attribute_ids, query_id=query_id)
            else:
                return self._reponse_attribute(item_ids=item_ids, attribute_ids=attribute_ids, query_id=query_id)
        elif query_type is QUERY_ATTRIBUTE_VAL_SIGNAL:
            query_attribute_id, query_attribute_vals = query_id
            return self._reponse_attribute_value(item_ids=item_ids, attribute_ids=attribute_ids, query_attribute_id=query_attribute_id, query_attribute_vals=query_attribute_vals)
        else:
            print(f"query type must be in {[QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL]}")
            raise NotImplementedError
    
    def evaluate(self, query_item_ids: List[int]):
        for query_item_id in query_item_ids:
            if query_item_id in self.label_ids:
                return SUCCESS_SIGNAL
        return FAILURE_SIGNAL
    
    def response_with_render(self, item_ids: List[int], attribute_ids: Dict[int, List[int]], query: str, num_turn: int) -> str:
        assert isinstance(self.render, BaseRender)
        query_type, query_id = self.render.query2ids(query=query, item_ids=item_ids, attribute_ids=attribute_ids)
        reponse_ids = self.response(item_ids=item_ids, attribute_ids=attribute_ids, query_type=query_type, query_id=query_id, num_turn=num_turn)
        return self.render.ids2response(response_ids=reponse_ids)