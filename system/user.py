import numpy as np

from typing import List

SUCCESS_SIGNAL = "success"
FAILURE_SIGNAL = "failure"


class UserAgent():
    def __init__(self, max_turn: int = 15, enable_not_know: bool = False, enabl_quit: bool = False):  
        self.max_turn = max_turn 
        self.enable_not_know = enable_not_know # enable user return Not Know
        self.enabl_quit = enabl_quit # enable user quit at max_turn
        # initial
        self.label_ids, self.data_matrix, self.label_data_matrix = None, None, None
    
    def set_session(self, session_id: int, data_matrix: np.ndarray, label_ids: List[int]):
        self.session_id = session_id
        # label_ids: a list of all the positive item ids in session
        self.label_ids = label_ids
        self.data_matrix = data_matrix
        self.label_data_matrix = data_matrix[label_ids,:]
    
    def _reponse_item(self, item_ids: List[int], attribute_ids: List[int], query_item_id: int):
        item_ids.remove(query_item_id)
        if query_item_id in self.label_ids:
            return "success"
        else:
            return (item_ids, attribute_ids)
    
    def _reponse_attribute(self, item_ids, attribute_ids, query_attribute_id):
        label_attribute_values = list(self.label_data_matrix[:,query_attribute_id])
        label_mask = np.isin(self.data_matrix[:,query_attribute_id], label_attribute_values) # select items equal to labels
        item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
        label_item_ids = item_ids_matrix[label_mask]
        item_ids = list(set(item_ids) & set(label_item_ids))
        attribute_ids.remove(query_attribute_id) # remove queried attributes
        return (item_ids, attribute_ids)
    
    def _reponse_attribute_with_not_know(self, item_ids, attribute_ids, query_attribute_id):
        label_attribute_values = list(set(self.label_data_matrix[:,query_attribute_id]))
        if len(label_attribute_values) >= 3:
            attribute_ids.remove(query_attribute_id)
            return (item_ids, attribute_ids)
        else:
            return self._reponse_attribute(item_ids, attribute_ids, query_attribute_id)
    
    def _reponse_attribute_value(self, item_ids, attribute_ids, attribute_vals, query_attribute_id, query_attribute_value):
        label_attribute_values = list(self.label_data_matrix[:,query_attribute_id])
        if query_attribute_value in label_attribute_values:
            label_mask = self.data_matrix[:,query_attribute_id] == query_attribute_value
            item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
            label_item_ids = item_ids_matrix[label_mask]
            item_ids = list(set(item_ids) & set(label_item_ids))
        else:
            not_label_mask = self.data_matrix[:,query_attribute_id] == query_attribute_value
            item_ids_matrix = np.array([_ for _ in range(self.data_matrix.shape[0])])
            not_label_item_ids = item_ids_matrix[not_label_mask]
            item_ids = list(set(item_ids) - set(not_label_item_ids))
        attribute_vals[query_attribute_id].remove(query_attribute_value) # remove queried value
        if not attribute_vals[query_attribute_id]:
            attribute_ids.remove(query_attribute_id) # if all the values are queried, remove attribute
        
        # only store label_ids in item_ids
        self.label_ids = list(set(self.label_ids) & set(item_ids))
        self.label_data_matrix = self.data_matrix[self.label_ids,:]
        return (item_ids, (attribute_ids, attribute_vals))

    def response(self, item_ids, attribute_ids, query_type, query_id, num_turn):
        if self.enabl_quit:
            if num_turn > self.max_turn:
                return "quit"
        if query_type is "item":
            return self._reponse_item(item_ids, attribute_ids, query_id)
        elif query_type is "attribute":
            if self.enable_not_know:
                return self._reponse_attribute_with_not_know(item_ids, attribute_ids, query_id)
            else:
                return self._reponse_attribute(item_ids, attribute_ids, query_id)
        elif query_type is "attribute_val":
            attribute_ids, attribute_vals = attribute_ids
            query_id, query_val = query_id
            return self._reponse_attribute_value(item_ids, attribute_ids, attribute_vals, query_id, query_val)
    
    def evaluate(self, query_item_id):
        if query_item_id in self.label_ids:
            return "success"
        else:
            return "failure"