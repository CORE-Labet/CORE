import numpy as np

from collections import Counter
from typing import List


class BaseChecker():
    def __init__(self, n_items: int = 1, n_attribute_val: int = 1, query_attribute_val: bool = False, 
                    query_attribute_only: bool = False, query_item_only: bool = False):
        self.n_items = n_items   # number of item shots
        self.n_attribute_val = n_attribute_val  # number of attribute value shots but all attribute values belong to one attribute
        self.query_attribute_val = query_attribute_val
        self.query_attribute_only = query_attribute_only
        self.query_item_only = query_item_only

    def act(self, item_ids: List[int], attribute_ids: List[int]):
        raise NotImplementedError


class ItemChecker(BaseChecker):
    def __init__(self, n_shots: int = 1, query_attribute_val: bool = False, query_attribute_only: bool = False, query_item_only: bool = False):
        super().__init__(n_shots, query_attribute_val, query_attribute_only, query_item_only)


class AttributeChecker(BaseChecker):
    def __init__(self, n_items: int = 1, n_attribute_val: int = 1, query_attribute_val: bool = False, query_attribute_only: bool = False, query_item_only: bool = False):
        super().__init__(n_items, n_attribute_val, query_attribute_val, query_attribute_only, query_item_only)
    

class Checker():
    def __init__(self, checker_name, config_dict, query_attribute_val=False):
        self.checker_name = checker_name
        self.config_dict = config_dict
        self.query_attribute_val = query_attribute_val
    
    def _calculate_query_item(self, data_matrix, item_ids):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = []
        for item_id in item_ids:
            prob = data_matrix[item_id][-1] / sum_score
            cert = (1 - prob) * data_matrix[item_id][-1] + prob * sum_score
            cert_gains.append(cert)
        max_cert = max(cert_gains)
        max_item_id = item_ids[cert_gains.index(max_cert)]
        return (max_item_id, max_cert)

    def _calculate_query_attribute(self, data_matrix, attribute_ids, dependence_helper=False):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = []
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            value_name = list(attribute_dict.keys())
            cert = 0
            for value in value_name:
                value_mask = data_matrix[:,attribute_id] != value # select items not equal to value
                value_score = data_matrix[value_mask][:,-1].sum()
                prob = 1 - value_score / sum_score
                cert += prob * value_score + (1 - prob) * (sum_score - value_score)
            cert_gains.append(cert)
        if dependence_helper:
            return cert_gains
        max_cert = max(cert_gains)
        max_attribute_id = attribute_ids[cert_gains.index(max_cert)]
        return (max_attribute_id, max_cert)
    
    def _calculate_query_attribute_value(self, data_matrix, attribute_ids, attribute_vals, dependence_helper=False):
        sum_score = data_matrix[:,-1].sum()
        cert_gains = [] # include gain of querying each attribute id
        cert_gain_vals = [] # include max attribute val of each attribute id
        for attribute_id in attribute_ids:
            assert attribute_vals[attribute_id]
            cert = []
            for value in attribute_vals[attribute_id]:
                value_mask = data_matrix[:,attribute_id] != value # select items not equal to value
                value_score = data_matrix[value_mask][:,-1].sum()
                prob = 1 - value_score / sum_score
                cert.append(prob * value_score + (1 - prob) * (sum_score - value_score))
            
            cert_gains.append(max(cert))
            cert_gain_vals.append(attribute_vals[attribute_id][cert.index(max(cert))])
        
        if dependence_helper:
            return (cert_gains, cert_gain_vals)
       
        max_cert = max(cert_gains)
        max_attribute_id = attribute_ids[cert_gains.index(max_cert)] # find max attribute id
        max_attribute_value = cert_gain_vals[cert_gains.index(max_cert)]
        # find max attribute value
        # for value in attribute_vals[max_attribute_id]:
        #     value_mask = data_matrix[:,max_attribute_id] != value # select items not equal to value
        #     value_score = data_matrix[value_mask][:,-1].sum()
        #     prob = 1 - value_score / sum_score
        #     cert = prob * value_score + (1 - prob) * (sum_score - value_score)
        #     if cert == max_cert:
        #         max_attribute_value = value
        return (max_attribute_id, max_attribute_value, max_cert)
    
    def _calculate_query_attribute_value_with_dependence(self, data_matrix, attribute_ids, attribute_vals, dependence_matrix=None):
        if self.config_dict["InsertDependence"]:
            assert dependence_matrix is not None
            cert_gains, cert_gain_vals = self._calculate_query_attribute_value(data_matrix, attribute_ids, attribute_vals, dependence_helper=True)
            cert_gains = np.dot(cert_gains, dependence_matrix)
            max_cert = max(cert_gains)
            max_attribute_id = attribute_ids[cert_gains.index(max_cert)]
            
            max_attribute_value = cert_gain_vals[cert_gains.index(max_cert)]
            # find max attribute value
            # sum_score = data_matrix[:,-1].sum()
            # for value in attribute_vals[max_attribute_id]:
            #     value_mask = data_matrix[:,max_attribute_id] != value # select items not equal to value
            #     value_score = data_matrix[value_mask][:,-1].sum()
            #     prob = 1 - value_score / sum_score
            #     cert = prob * value_score + (1 - prob) * (sum_score - value_score)
            #     if cert == max_cert:
            #         max_attribute_value = value
            return (max_attribute_id, max_attribute_value, max_cert)
        else:
            raise NotImplementedError
    
    def _calculate_query_attribute_with_dependence(self, data_matrix, attribute_ids, dependence_matrix=None):
        # use FM-based method to generate dependence weights, shape [attribute_num, attribute_num]
        if self.config_dict["InsertDependence"]:
            assert dependence_matrix is not None
            cert_gains = self._calculate_query_attribute(data_matrix, attribute_ids, dependence_helper=True)
            cert_gains = np.dot(cert_gains, dependence_matrix)
            max_cert = max(cert_gains)
            max_attribute_id = attribute_ids[cert_gains.index(max_cert)]
            return (max_attribute_id, max_cert)
        else:
            raise NotImplementedError
    
    def _act_core_checker(self, data_matrix, item_ids, attribute_ids, num_turn=0, is_evaluate=False):
        if self.value_querier:
            attribute_ids, attribute_vals = attribute_ids
        max_item_id, max_item_cert = self._calculate_query_item(data_matrix, item_ids)
        if self.config_dict["ItemOnly"] or is_evaluate:
            return ("item", max_item_id)
        candidate_data_matrix = data_matrix[item_ids] # data matrix for unchecked items
        if self.config_dict["AttributeDependence"]:
            if self.value_querier:
                max_attribute_id, max_attribute_value, max_attribute_cert = self._calculate_query_attribute_value_with_dependence(candidate_data_matrix, attribute_ids, attribute_vals)
            else:
                max_attribute_id, max_attribute_cert = self._calculate_query_attribute_with_dependence(candidate_data_matrix, attribute_ids)
        else:
            if self.value_querier:
                max_attribute_id, max_attribute_value, max_attribute_cert = self._calculate_query_attribute_value(candidate_data_matrix, attribute_ids, attribute_vals)  
            else:
                max_attribute_id, max_attribute_cert = self._calculate_query_attribute(candidate_data_matrix, attribute_ids)
        if self.config_dict["AttributeOnly"]:
            if self.value_querier:
                return ("attribute_val", (max_attribute_id, max_attribute_value))
            else:
                return ("attribute", max_attribute_id)
        if max_item_cert >= max_attribute_cert:
            return ("item", max_item_id)
        else:
            if self.config_dict["TurnPenalty"]:
                if max_item_cert + self.config_dict["TurnPenaltyWeight"] * num_turn > max_attribute_cert:
                    return ("item", max_item_id)
            else:
                if self.value_querier:
                    return ("attribute_val", (max_attribute_id, max_attribute_value))
                else:
                    return ("attribute", max_attribute_id)
    
    def _act_max_entropy_attribute_checker(self, data_matrix, attribute_ids):
        entros = []
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            value_num = list(attribute_dict.values())
            sum_value = sum(value_num)
            entro = 0 # entropy for attribute_id
            for value in value_num:
                entro += - (value / sum_value) * np.log2(value / sum_value)
            entros.append(entro)
        max_attribute_id = attribute_ids[entros.index(max(entros))]
        return ("attribute", max_attribute_id)
    
    def _act_greedy_item_checker(self, data_matrix, item_ids):
        max_score, max_item_id = 0, 0
        
        for item_id in item_ids:
            if data_matrix[item_id][-1] > max_score:
                max_score = data_matrix[item_id][-1]
                max_item_id = item_id
        return ("item", max_item_id)
    
    def act(self, data_matrix, item_ids, attribute_ids, num_turn):
        if self.checker_name == "core":
            return self._act_core_checker(data_matrix, item_ids, attribute_ids, num_turn=num_turn)
        elif self.checker_name == "attribute":
            if attribute_ids:
                return self._act_max_entropy_attribute_checker(data_matrix, attribute_ids)
            else:
                return self._act_greedy_item_checker(data_matrix, item_ids)
        elif self.checker_name == "item":
            return self._act_greedy_item_checker(data_matrix, item_ids)
        else:
            raise NotImplementedError
    
    def evaluate(self, data_matrix, item_ids, attribute_ids):
        if self.checker_name == "core":
            return self._act_core_checker(data_matrix, item_ids, attribute_ids, is_evaluate=True)
        elif self.checker_name == "attribute" or "item":
            return self._act_greedy_item_checker(data_matrix, item_ids)
        else:
            raise NotImplementedError