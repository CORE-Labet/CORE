import numpy as np
from typing import List


class DataManager:
    def __init__(self, data_matrx: np.ndarray):
        self.data_matrix = data_matrx
        self.item_ids = []
        self.attribute_ids = {}
        self.label_ids = []
    
    def store_data(self, data_matrx: np.ndarray):
        self.data_matrix = data_matrx
    
    def set_session(self, candidate_item_ids: List):
        self.label_ids = self.data_matrix[:-1]
    
    def set_turn(self, response, query_type, query_id):
        pass

    def get_checked_item_ids(self):
        pass

    def get_checked_attribute_ids(self):
        pass

    def update_labels(self):
        pass