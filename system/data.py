import numpy as np

from typing import List
from user import QUIT_SINGAL, NOT_KNOW_SINGAL


class DataManager:
    def __init__(self, data_matrx: np.ndarray):
        self.data_matrix = data_matrx
        self.item_ids = []
        self.attribute_ids = {}
        self.label_ids = []
    
    def set_unique_data_map(self):
        pass
    
    def store_data(self, data_matrx: np.ndarray):
        self.data_matrix = data_matrx
    
    def set_session(self, session_id: int):
        self.label_ids = self.data_matrix[:-1]
    
    def set_turn(self, response, query_type, query_id):
        pass

    def get_checked_item_ids(self):
        pass

    def get_checked_attribute_ids(self):
        pass

    def update_labels(self):
        pass