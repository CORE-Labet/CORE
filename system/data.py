import numpy as np
import random

from typing import List, Dict
from collections import Counter
from retriever import BaseRetriever
from user import QUIT_SINGAL, NOT_KNOW_SINGAL


class DataManager:
    def __init__(self, retriever: BaseRetriever):
        self.label_ids = {}
        self.user_ids = []
        self.item_ids = []
        self.retriever = retriever

        self.session_id = 0
        self.turn_id = 0
        self.session2user = {}  # map from session_id to user_id
        self.candidate_data_matrix = None
    
    def set_num_candidate_items(self, num_candidate_item_per_session: int):
        self.retriever.num_candidate_items = num_candidate_item_per_session
    
    def reset4simulation(self, train_data_matrix: np.ndarray, train_label_ids: Dict[int, List[int]], test_data_matrix: np.ndarray,
                                    test_label_ids: Dict[int, List[int]]): # to simulate online checking and offline training
        self.train_label_ids = train_label_ids
        self.test_label_ids = test_label_ids
        self.session_id = 0
        self.session2user = {}

        assert len(train_data_matrix.shape) == len(test_data_matrix.shape) and train_data_matrix.shape[1] == test_data_matrix.shape[1], f"train_data_matrix {train_data_matrix.shape} not match test_data_matrix {test_data_matrix.shape}"
        data_matrix = np.stack((train_data_matrix, test_data_matrix), axis=0)
        candidate_data_matrix = data_matrix[:,1:]
        item_id_col = candidate_data_matrix[:,0] 
        unique_mask = np.unique(item_id_col, return_index=True)[1]  # keep col unique
        self.candidate_data_matrix = candidate_data_matrix[unique_mask]

    def set_session4simulation(self, user_id: int = None, enable_overlap: bool = False):   # to simulate online checking and offline training
        assert self.user_ids and self.item_ids, "load data first"
        if not user_id:
            if not enable_overlap:
                checked_user_ids = list(self.session2user.values)
                user_ids = [idx for idx in self.test_label_ids.keys() if idx not in checked_user_ids]
            else:
                user_ids = list(self.test_label_ids.keys())
            user_id = random.choice(user_ids)
        else:
            user_id = user_id
        assert self.test_label_ids[user_id], f"user_id {user_id} is not valid"



        candidate_item_ids = self.test_label_ids[user_id]
        


    
    def reset(self, data_matrix: np.ndarray, label_ids: Dict[int, List[int]]):
        self.label_ids = label_ids
        self.session_id = 0
        self.session2user = {}
        
        user_ids = data_matrix[:0].tolist()    # user_ids are stored at 1st col
        self.user_ids = list(set(user_ids)).sort()
        item_ids = data_matrix[:1].tolist()    # item_ids are stored at 2nd col
        self.item_ids = list(set(item_ids)).sort()

        candidate_data_matrix = data_matrix[:,1:]
        item_id_col = candidate_data_matrix[:,0] 
        unique_mask = np.unique(item_id_col, return_index=True)[1]  # keep col unique
        self.candidate_data_matrix = candidate_data_matrix[unique_mask]
    
    def set_session(self, user_id: int = None, enable_overlap: bool = False):
        assert self.user_ids and self.item_ids, "load data first"

        self.session_id += 1
        self.turn_id = 0
        if not user_id:
            if not enable_overlap:
                checked_user_ids = list(self.session2user.values)
                user_ids = [idx for idx in self.user_ids if idx not in checked_user_ids]
            else:
                user_ids = self.user_ids
            user_id = random.choice(user_ids)
        else:
            user_id = user_id
        assert self.label_ids[user_id], f"user_id {user_id} is not valid"

        candidate_item_ids = self.retriever.sample(item_ids=self.item_ids)
        session_mask = self.candidate_data_matrix[:,0] in candidate_item_ids
        candidate_data_matrix = self.candidate_data_matrix[session_mask]

        candidate_label_item_ids = list(set(self.label_ids[user_id] & candidate_item_ids))
        return (user_id, candidate_item_ids, candidate_data_matrix, candidate_label_item_ids)

    def store(self, new_data_matrix: np.ndarray, new_label_ids: Dict[int, List[int]]):
        assert len(new_data_matrix.shape) == len(self.data_matrix.shape) and new_data_matrix.shape[1] == self.data_matrix.shape[1], f"new_data_matrix {new_data_matrix.shape} not match data_matrix {self.data_matrix.shape}"
        self.data_matrix = np.stack((self.data_matrix, new_data_matrix), axis=0)
        
        for user_id, label_ids in new_label_ids.items():
            if user_id in self.label_ids:
                self.label_ids[user_id].extend(label_ids)
            else:
                self.label_ids[user_id] = label_ids
        self.reset()