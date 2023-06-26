import numpy as np
import random

from typing import List, Dict
from collections import Counter
from retriever import BaseRetriever


class DataManager():
    def __init__(self, retriever: BaseRetriever):
        self.label_ids = {}
        self.user_ids = []
        self.item_ids = []
        self.retriever = retriever
        
        # online checking
        self.session_id = 0
        self.turn_id = 0
        self.session2user = {}  # map from session_id to user_id
        self.all_user_matrix, self.all_item_matrix, self.all_interaction_matrix = None, None, None
        self.candidate_user_matrix, self.candidate_item_matrix, self.candidate_interaction_matrix = None, None, None
    
    def set_retriever(self, num_candidate_items: int = None, pos_neg_ratio: float = None):
        self.retriever.num_candidate_items = num_candidate_items if num_candidate_items else self.retriever.num_candidate_items
        self.retriever.pos_neg_ratio = pos_neg_ratio if pos_neg_ratio else self.retriever.pos_neg_ratio
    
    def check(self):
        assert len(self.user_ids) == self.all_user_matrix.shape[0] and len(self.item_ids) == self.all_item_matrix.shape[0]
        assert self.all_interaction_matrix.shape[-1] == 3
    
    def load(self, user_matrix: np.ndarray, item_matrix: np.ndarray, interaction_matrix: np.ndarray):
        self.all_user_matrix = user_matrix
        self.all_item_matrix = item_matrix
        self.all_interaction_matrix = interaction_matrix
        
        user_ids = user_matrix[:0].tolist()    # user_ids are stored at 1st col
        self.user_ids = list(set(user_ids)).sort()
        item_ids = item_matrix[:1].tolist()    # item_ids are stored at 2nd col
        self.item_ids = list(set(item_ids)).sort()

        self.label_ids = {}
        for user_id in self.user_ids:
            mask = (interaction_matrix[:, 0] == user_id) & (interaction_matrix[:, -1] >= 1)
            self.label_ids[user_id] = list(set(interaction_matrix[mask][:,1].tolist()))
        self.check()


    def store(self, new_user_matrix: np.ndarray = None, new_item_matrix: np.ndarray = None, new_interaction_matrix: np.ndarray = None):
        assert len(self.all_user_matrix.shape) == len(new_user_matrix.shape) and self.all_user_matrix.shape[1] == new_user_matrix.shape[1], f"new_user_matrix {new_user_matrix.shape} not match all_user_matrix {self.all_user_matrix.shape}"
        assert len(self.all_item_matrix.shape) == len(new_item_matrix.shape) and self.all_item_matrix.shape[1] == new_item_matrix.shape[1], f"new_item_matrix {new_item_matrix.shape} not match all_item_matrix {self.all_item_matrix.shape}"
        assert len(self.all_interaction_matrix.shape) == len(new_interaction_matrix.shape) and self.all_interaction_matrix.shape[1] == new_interaction_matrix.shape[1], f"new_interaction_matrix {new_interaction_matrix.shape} not match all_interaction_matrix {self.all_interaction_matrix.shape}"
        self.all_user_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)
        self.all_item_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)
        self.all_interaction_matrix = np.stack((self.all_interaction_matrix, new_interaction_matrix), axis=0)
        
        user_ids = self.all_user_matrix[:0].tolist()    # user_ids are stored at 1st col
        self.user_ids = list(set(user_ids)).sort()
        item_ids = self.all_item_matrix[:1].tolist()    # item_ids are stored at 2nd col
        self.item_ids = list(set(item_ids)).sort()
        
        self.label_ids = {}
        for user_id in self.user_ids:
            mask = (self.all_interaction_matrix[:, 0] == user_id) & (self.all_interaction_matrix[:, -1] >= 1)
            self.label_ids[user_id] = list(set(self.all_interaction_matrix[mask][:,1].tolist()))
        self.check()
    
    def setup4online_checker(self):
        pass

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
        if user_id not in self.train_label_ids:
            self.train_label_ids[user_id] = []

        pos_item_ids = self.test_label_ids[user_id]
        neg_item_ids = [idx for idx in self.item_ids if idx not in self.train_label_ids[user_id] and idx not in self.test_label_ids[user_id]]
        candidate_item_ids = self.retriever.sample_with_ratio(pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids)
        session_mask = self.candidate_data_matrix[:,0] in candidate_item_ids
        candidate_data_matrix = self.candidate_data_matrix[session_mask]

        candidate_label_item_ids = list(set(self.test_label_ids[user_id] & candidate_item_ids))
        candidate_label_attribute_ids = self._compute_label_attribute_ids(data_matrix=candidate_data_matrix, label_ids=candidate_label_item_ids)
        return (candidate_data_matrix, candidate_label_item_ids, candidate_label_attribute_ids)

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

        pos_item_ids = self.label_ids[user_id]
        neg_item_ids = [idx for idx in self.item_ids if idx not in pos_item_ids]
        candidate_item_ids = self.retriever.sample_with_ratio(pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids)
        session_mask = self.candidate_data_matrix[:,0] in candidate_item_ids
        candidate_data_matrix = self.candidate_data_matrix[session_mask]

        candidate_label_item_ids = list(set(self.label_ids[user_id] & candidate_item_ids))
        return (user_id, candidate_item_ids, candidate_data_matrix, candidate_label_item_ids)

    def _compute_label_attribute_ids(self, data_matrix: np.ndarray, label_ids: List[int]): 
        label_data_matrix = data_matrix[data_matrix[:0] in label_ids]
        attribute_ids = [_ for _ in range(1, label_data_matrix.shape[1])]  # 1st is item_id
        label_attribute_ids = {}
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            label_attribute_ids.update({attribute_id: list(attribute_dict.keys())})
        return label_attribute_ids
