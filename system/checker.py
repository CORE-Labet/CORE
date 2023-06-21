import numpy as np
import random

from collections import Counter
from typing import List, Dict
from agent import QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL


class BaseChecker():
    def __init__(self, n_items: int, n_attribute_val: int, query_attribute_val: bool, 
                    query_attribute_only: bool, query_item_only: bool):
        self.n_items = n_items   # number of item shots
        self.n_attribute_val = n_attribute_val  # number of attribute value shots but all attribute values belong to one attribute
        self.query_attribute_val = query_attribute_val
        self.query_attribute_only = query_attribute_only
        self.query_item_only = query_item_only
    
    def _sort_item(self, data_matrix: np.ndarray, item_ids: List[int]):
        score_dict = dict(zip(item_ids, data_matrix[item_ids,-1].tolist()))
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
        return list(score_dict.keys())

    def act(self, data_matrix: np.ndarray, item_ids: List[int], attribute_ids=Dict[int: List[int]], num_turn=0):
        raise NotImplementedError

    def evaluate(self, data_matrix: np.ndarray, item_ids: List[int]):
        raise NotImplementedError


class ItemChecker(BaseChecker):
    def __init__(self, n_items: int = 1, n_attribute_val: int = 1, query_attribute_val: bool = False, 
                    query_attribute_only: bool = False, query_item_only: bool = False):
        super().__init__(n_items=n_items, n_attribute_val=n_attribute_val, query_attribute_val=query_attribute_val, 
                            query_attribute_only=query_attribute_only, query_item_only=query_item_only)

    def act(self, data_matrix: np.ndarray, item_ids: List[int], attribute_ids=Dict[int: List[int]], num_turn=0):
        assert not self.query_attribute_only
        sorted_items = self._sort_item(data_matrix=data_matrix, item_ids=item_ids)
        return (QUERY_ITEM_SIGNAL, sorted_items[:self.n_items])

    def evaluate(self, data_matrix: np.ndarray, item_ids: List[int]):
        self.act(data_matrix=data_matrix, item_ids=item_ids)


class AttributeChecker(BaseChecker):
    def __init__(self, n_items: int = 1, n_attribute_val: int = 1, query_attribute_val: bool = False, 
                    query_attribute_only: bool = False, query_item_only: bool = False):
        super().__init__(n_items=n_items, n_attribute_val=n_attribute_val, query_attribute_val=query_attribute_val, 
                            query_attribute_only=query_attribute_only, query_item_only=query_item_only)

    def act(self, data_matrix: np.ndarray, item_ids: List[int], attribute_ids=Dict[int: List[int]], num_turn=0):
        assert not self.query_item_only
        entros = []
        for attribute_id in attribute_ids.keys():
            num_dict = Counter(data_matrix[:,attribute_id])
            value_num = list(num_dict.values())
            sum_value = sum(value_num)
            entro = 0 # entropy for attribute_id
            for value in value_num:
                entro += - (value / sum_value) * np.log2(value / sum_value)
            entros.append(entro)
        max_attribute_id = list(attribute_ids.keys())[entros.index(max(entros))]

        if self.query_attribute_val:
            random_attribute_values = random.choice(attribute_ids[max_attribute_id], k=self.n_attribute_val)     # randomly select a value
            return (QUERY_ATTRIBUTE_VAL_SIGNAL, (max_attribute_id, random_attribute_values))
        else:
            return (QUERY_ATTRIBUTE_SINGAL, max_attribute_id)
    
    def evaluate(self, data_matrix: np.ndarray, item_ids: List[int]):
        sorted_items = self._sort_item(data_matrix=data_matrix, item_ids=item_ids)
        return (QUERY_ITEM_SIGNAL, sorted_items[:self.n_items])
    

class CoreChecker(BaseChecker):
    def __init__(self, n_items: int = 1, n_attribute_val: int = 1, query_attribute_val: bool = False, 
                    query_attribute_only: bool = False, query_item_only: bool = False, 
                    enable_penalty: bool = False, penalty_weight: float = 0.0):
        super().__init__(n_items=n_items, n_attribute_val=n_attribute_val, query_attribute_val=query_attribute_val, 
                            query_attribute_only=query_attribute_only, query_item_only=query_item_only)
        self.enable_penalty = enable_penalty
        self.penalty_weight = penalty_weight
        self.enable_dependence = False
        self.dependence_matrx = None
    
    def load(self, enable_dependence: bool = False, dependence_matrx: np.ndarray = None):
        self.enable_dependence = enable_dependence
        self.dependence_matrx = dependence_matrx
    
    def _calculate_item(self, data_matrix: np.ndarray, item_ids: List[int]):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = []
        for item_id in item_ids:
            prob = data_matrix[item_id][-1] / sum_score
            cert = (1 - prob) * data_matrix[item_id][-1] + prob * sum_score
            cert_gains.append(cert)
        score_dict = dict(zip(item_ids, cert_gains))
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
        
        max_item_ids = list(score_dict.keys())[:self.n_items]
        max_cert = sum(list(score_dict.values())[:self.n_items])
        return (max_item_ids, max_cert)

    def _calculate_attribute(self, data_matrix: np.ndarray, attribute_ids: Dict[int, List]):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = []
        for attribute_id in attribute_ids.keys():
            num_dict = Counter(data_matrix[:,attribute_id])
            value_name = list(num_dict.keys())
            cert = 0
            for value in value_name:
                value_mask = data_matrix[:,attribute_id] != value # select items not equal to value
                value_score = data_matrix[value_mask][:,-1].sum()
                prob = 1 - value_score / sum_score
                cert += prob * value_score + (1 - prob) * (sum_score - value_score)
            cert_gains.append(cert)
        max_cert = max(cert_gains)
        max_attribute_id = attribute_ids[cert_gains.index(max_cert)].keys()
        return (max_attribute_id, max_cert)

    def _calculate_attribute_with_dependence(self):
        raise NotImplementedError
    
    def _calculate_attribute_val(self, data_matrix: np.ndarray, attribute_ids: List[Dict[int, List]]):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = {} # include attribute_id: gain of querying each attribute id (sum of top-n_attribute_vals)
        attribute_vals = {} # include attribute_id: top-n_attribute_vals attribute vals of attribute id
        for attribute_id in attribute_ids:
            vals, cert = [], []
            for value in attribute_id:
                value_mask = data_matrix[:,attribute_id.keys()] != value # select items not equal to value
                value_score = data_matrix[value_mask][:,-1].sum()
                prob = 1 - value_score / sum_score
                cert.append(prob * value_score + (1 - prob) * (sum_score - value_score)) 
                vals.append(value)
            cert_dict = dict(zip(vals, cert))
            cert_dict = dict(sorted(cert_dict.items(), key=lambda x: x[1], reverse=True))
            cert_gains.update({attribute_id: sum(cert_dict.values()[:self.n_attribute_val])})
            attribute_vals.update({attribute_id: cert_dict.keys()[:self.n_attribute_val]})
        
        max_attribute_id = max(cert_gains, key=cert_gains.get)
        max_cert = cert_gains[max_attribute_id]
        max_attribute_vals = attribute_vals[max_attribute_id]
        return (max_attribute_id, max_attribute_vals, max_cert)
    
    def _calculate_attribute_val_with_dependence(self):
        raise NotImplementedError
    
    def act(self, data_matrix: np.ndarray, item_ids: List[int], attribute_ids=Dict[int: List[int]], turn_id=0):
        if self.query_item_only:
            max_item_ids, _ = self._calculate_item(data_matrix=data_matrix, item_ids=item_ids)
            return (QUERY_ITEM_SIGNAL, max_item_ids)
        if self.query_attribute_only:
            if self.query_attribute_val:
                max_attribute_id, max_attribute_vals, _ = self._calculate_attribute_val(data_matrix=data_matrix, attribute_ids=attribute_ids)
                return (QUERY_ATTRIBUTE_VAL_SIGNAL, (max_attribute_id, max_attribute_vals))
            else:
                max_attribute_id, _ = self._calculate_attribute(data_matrix=data_matrix, attribute_ids=attribute_ids)
                return (QUERY_ATTRIBUTE_SINGAL, max_attribute_id)
        
        max_item_ids, max_item_cert = self._calculate_item(data_matrix=data_matrix, item_ids=item_ids)
        if self.query_attribute_val:
            max_attribute_id, max_attribute_vals, max_value_cert = self._calculate_attribute_val(data_matrix=data_matrix, attribute_ids=attribute_ids)
            if max_item_cert >= max_value_cert:
                return (QUERY_ITEM_SIGNAL, max_item_ids)
            else:
                if self.enable_penalty and self.penalty_weight and self.penalty_weight * turn_id >= max_value_cert:
                    return (QUERY_ITEM_SIGNAL, max_item_ids)
                else:
                    return (QUERY_ATTRIBUTE_VAL_SIGNAL, (max_attribute_id, max_attribute_vals))
        else:
            max_attribute_id, max_value_cert = self._calculate_attribute(data_matrix=data_matrix, attribute_ids=attribute_ids)
            if max_item_cert >= max_value_cert:
                return (QUERY_ITEM_SIGNAL, max_item_ids)
            else:
                if self.enable_penalty and self.penalty_weight and self.penalty_weight * turn_id >= max_value_cert:
                    return (QUERY_ITEM_SIGNAL, max_item_ids)
                else:
                    return (QUERY_ATTRIBUTE_SINGAL, max_attribute_id)

    def evaluate(self, data_matrix: np.ndarray, item_ids: List[int]):
        max_item_ids, _ = self._calculate_item(data_matrix=data_matrix, item_ids=item_ids)
        return (QUERY_ITEM_SIGNAL, max_item_ids)