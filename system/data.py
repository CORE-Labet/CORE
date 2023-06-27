import numpy as np
import random

from torch.utils.data import Dataset
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
        self.session2user = {}  # map from session_id to user_id
        self.all_user_matrix, self.all_item_matrix, self.all_interaction_matrix = None, None, None  # data for storage
        self.online_user_ids = []   # data for online checker
    
    def set_retriever(self, num_candidate_items: int = None, pos_neg_ratio: float = None):
        self.retriever.num_candidate_items = num_candidate_items if num_candidate_items else self.retriever.num_candidate_items
        self.retriever.pos_neg_ratio = pos_neg_ratio if pos_neg_ratio else self.retriever.pos_neg_ratio
    
    def check(self):
        assert len(self.user_ids) == self.all_user_matrix.shape[0] and len(self.item_ids) == self.all_item_matrix.shape[0]
        assert self.all_interaction_matrix.shape[-1] == 3

        # use the lastest feature matrix of user_id and item_id to represent
        user_id_col = self.all_user_matrix[:,0]
        unique_mask = np.unique(user_id_col, return_index=True)[1]
        self.all_user_matrix = self.all_user_matrix[unique_mask]

        item_id_col = self.all_item_matrix[:,0] 
        unique_mask = np.unique(item_id_col, return_index=True)[1] 
        self.all_item_matrix = self.all_item_matrix[unique_mask]
    
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
        if new_user_matrix:
            assert len(self.all_user_matrix.shape) == len(new_user_matrix.shape) and self.all_user_matrix.shape[1] == new_user_matrix.shape[1], f"new_user_matrix {new_user_matrix.shape} not match all_user_matrix {self.all_user_matrix.shape}"
            self.all_user_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)
            user_ids = self.all_user_matrix[:0].tolist()    # user_ids are stored at 1st col
            self.user_ids = list(set(user_ids)).sort()
        if new_item_matrix:
            assert len(self.all_item_matrix.shape) == len(new_item_matrix.shape) and self.all_item_matrix.shape[1] == new_item_matrix.shape[1], f"new_item_matrix {new_item_matrix.shape} not match all_item_matrix {self.all_item_matrix.shape}"
            self.all_item_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)
            item_ids = self.all_item_matrix[:1].tolist()    # item_ids are stored at 2nd col
            self.item_ids = list(set(item_ids)).sort()
        
        if new_interaction_matrix:
            assert len(self.all_interaction_matrix.shape) == len(new_interaction_matrix.shape) and self.all_interaction_matrix.shape[1] == new_interaction_matrix.shape[1], f"new_interaction_matrix {new_interaction_matrix.shape} not match all_interaction_matrix {self.all_interaction_matrix.shape}"
            self.all_interaction_matrix = np.stack((self.all_interaction_matrix, new_interaction_matrix), axis=0)
        
            self.label_ids = {}
            for user_id in self.user_ids:
                mask = (self.all_interaction_matrix[:, 0] == user_id) & (self.all_interaction_matrix[:, -1] >= 1)
                self.label_ids[user_id] = list(set(self.all_interaction_matrix[mask][:,1].tolist()))
        
        if new_user_matrix or new_item_matrix or new_interaction_matrix:
            self.check()
    
    def set_online_checker(self, online_label_ids: Dict[int, List[int]] = None, split_ratio: float = 0.8):
        if online_label_ids:
            candidate_user_ids = list(online_label_ids.keys())
            assert not [user_id for user_id in candidate_user_ids if user_id not in self.user_ids]
            self.online_user_ids = candidate_user_ids
        else:
            if self.online_user_ids:
                candidate_user_ids = self.online_user_ids
            else:
                candidate_user_ids = random.choices(self.user_ids, k=int((1-split_ratio)*len(self.user_ids)))
                self.online_user_ids = candidate_user_ids
            
            online_label_ids = {}
            for user_id in self.online_user_ids:
                online_label_ids[user_id] = self.label_ids[user_id]
        
        self.session_id = 0
        self.session2user = {}
        return len(self.online_user_ids)
    
    def set_offline_trainer(self, split_ratio: float = 0.8):
        if self.online_user_ids:
            candidate_user_ids = [user_id for user_id in self.user_ids if user_id not in self.online_user_ids]
        else:
            candidate_user_ids = random.choices(self.user_ids, k=int(split_ratio*len(self.user_ids)))
            self.online_user_ids = [user_id for user_id in self.user_ids if user_id not in candidate_user_ids]
        
        class Dataset4Train(Dataset):
            def __init__(self) -> None:
                super().__init__()

    def set_session(self, user_id: int = None, enable_overlap: bool = False):   # to simulate online checking and offline training
        assert self.online_user_ids and self.user_ids and self.item_ids and self.label_ids, "load data and setup online checker first"
        if not user_id:
            if not enable_overlap:
                checked_user_ids = list(self.session2user.values())
                user_ids = [user_id for user_id in self.online_user_ids if user_id not in checked_user_ids]
            else:
                user_ids = self.online_user_ids
            user_id = random.choice(user_ids)
        else:
            if user_id not in self.online_user_ids:
                self.online_user_ids.append(user_id)
        assert self.label_ids[user_id], f"user_id {user_id} is not valid"
        
        self.session2user[self.session_id] = user_id
        self.session_id += 1

        pos_item_ids = self.label_ids[user_id]
        neg_item_ids = [idx for idx in self.item_ids if idx not in pos_item_ids]
        candidate_item_ids = self.retriever.sample_with_ratio(pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids)
        session_mask = self.candidate_data_matrix[:,0] in candidate_item_ids
        candidate_data_matrix = self.candidate_data_matrix[session_mask]

        candidate_label_item_ids = list(set(self.label_ids[user_id] & candidate_item_ids))
        candidate_label_attribute_ids = self._compute_label_attribute_ids(data_matrix=candidate_data_matrix, label_ids=candidate_label_item_ids)
        return (candidate_data_matrix, candidate_label_item_ids, candidate_label_attribute_ids)

    def _compute_label_attribute_ids(self, data_matrix: np.ndarray, label_ids: List[int]): 
        label_data_matrix = data_matrix[data_matrix[:0] in label_ids]
        attribute_ids = [_ for _ in range(1, label_data_matrix.shape[1])]  # 1st is item_id
        label_attribute_ids = {}
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            label_attribute_ids.update({attribute_id: list(attribute_dict.keys())})
        return label_attribute_ids
