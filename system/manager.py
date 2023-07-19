import numpy as np
import random
import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from collections import Counter
from retriever import BaseRetriever
from trainer import BaseTrainer


# organize data for offline trainer
class Dataset4Trainer(Dataset):
    def __init__(self, seq_data: Dict, seq_len: Dict, split_ratio: Dict):
        super(Dataset4Trainer, self).__init__()
        assert set(seq_data.keys()) == set(seq_len.keys()), "user_id in seq_data and seq_len should be same"
        self.user_ids = list(seq_data.keys())
        self.user_ids.sort()
        
        self.seq_data = seq_data
        self.seq_len = seq_len

        self.split_ratio = split_ratio
        self.split_locs = None
                    
    def __len__(self):
        return len(self.seq_data)
    
    def set_mode(self, mode: str):
        assert mode in self.split_ratio.keys()
        self.split_locs = self.split_ratio[mode]

    def __getitem__(self, idx):
        assert self.split_locs, "please set mode first: train or valid"
        user_id = self.user_ids[idx]
        seq_data = self.seq_data[user_id]
        seq_len = self.seq_len[user_id]
        begin, end = int(seq_len * self.split_locs[0]), int(seq_len * self.split_locs[1])
        x = seq_data[:end,:-1] # feat
        y = seq_data[begin:end,-1]  # label
        return (x, y)


# organize session data for online checker
class Dataset4Checker(Dataset):
    def __init__(self, seq_data: list, seq_len: int):
        super(Dataset4Checker, self).__init__()
        self.seq_data = seq_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seq_data)
    
    def __getitem__(self, idx):
        begin, end = self.seq_len, self.seq_len + 1
        seq_data = self.seq_data[idx]
        x = seq_data[:end,:-1]  # feat
        y = seq_data[begin:end,-1]  # label
        return (x, y)


class DataManager():
    def __init__(self, data_name: str, data_path: str = None, retriever: BaseRetriever = None, 
                        score_func: str = "embedding", pad_len: int = 8):
        self.data_name = data_name
        if not data_path:
            assert data_name, "at least one of data_path and data_name should be vaild"
            current_path = os.path.abspath(os.getcwd())
            self.data_path = os.path.join(os.path.dirname(current_path), "data/")
        else:
            self.data_path = data_path  # text data storage
        
        self.label_ids = {}
        self.user_ids = []
        self.item_ids = []
        self.retriever = retriever

        # online checking
        self.session_id = 0
        self.session2user = {}  # map from session_id to user_id
        self.all_user_matrix, self.all_item_matrix, self.all_interaction_matrix = None, None, None  # data for storage
        self.online_user_ids = []   # include all the online_user_ids
        self.score_func = score_func

        # offline training
        self.pad_len = pad_len

    def set_retriever(self, num_candidate_items: int = None, pos_neg_ratio: float = None):
        self.retriever.num_candidate_items = num_candidate_items if num_candidate_items else self.retriever.num_candidate_items
        self.retriever.pos_neg_ratio = pos_neg_ratio if pos_neg_ratio else self.retriever.pos_neg_ratio
    
    def check(self):
        # use the lastest feature matrix of user_id and item_id to represent
        user_id_col = self.all_user_matrix[:,0]
        unique_mask = np.unique(user_id_col, return_index=True)[1]
        self.all_user_matrix = self.all_user_matrix[unique_mask]

        item_id_col = self.all_item_matrix[:,0] 
        unique_mask = np.unique(item_id_col, return_index=True)[1] 
        self.all_item_matrix = self.all_item_matrix[unique_mask]

        assert set(self.user_ids) == set(self.all_user_matrix[:,0]) 
        assert set(self.item_ids) == set(self.all_item_matrix[:,0])
        assert set(self.all_interaction_matrix[:,0]).issubset(set(self.user_ids)), "each user should have feat"
        assert set(self.all_interaction_matrix[:,1]).issubset(set(self.item_ids)), "each item should have feat"
    
    def get_statics(self):
        # num_values includes all uniques values in all cols
        num_val = 0
        for col in self.all_user_matrix.T:  # col in user side
            num_val += len(set(col))
        for col in self.all_item_matrix.T:  # col in item side
            num_val += len(set(col))
        if self.all_interaction_matrix.shape[1] > 3:    # col in interaction (both sides)
            for col in self.all_interaction_matrix[:,4:].T:
                num_val += len(set(col))
        num_feat = self.all_user_matrix.shape[1] + self.all_item_matrix.shape[1]
        return (num_val, num_feat)
    
    def load(self, user_matrix: np.ndarray = None, item_matrix: np.ndarray = None, interaction_matrix: np.ndarray = None):
        if not (user_matrix and item_matrix and interaction_matrix):
            assert os.path.exists(os.path.join(self.data_path, f"{self.data_name}.pickle")), f"load data from file {self.data_path}" + f"{self.data_name}.pickle"
            with open(os.path.join(self.data_path, f"{self.data_name}.pickle"), "rb") as f:   
                user_matrix, item_matrix, interaction_matrix = pickle.load(f)

        self.all_user_matrix = user_matrix
        self.all_item_matrix = item_matrix
        self.all_interaction_matrix = interaction_matrix
        
        user_ids = user_matrix[:,0].tolist()    # user_ids are stored at 1st col
        self.user_ids = list(set(user_ids))
        self.user_ids.sort()
        item_ids = item_matrix[:,0].tolist()    # item_ids are stored at 1st col
        self.item_ids = list(set(item_ids))
        self.item_ids.sort()

        self.label_ids = {}
        for user_id in self.user_ids:
            mask = (interaction_matrix[:,0] == user_id) & (interaction_matrix[:,-1] >= 1)
            self.label_ids[user_id] = list(set(interaction_matrix[mask][:,1].tolist()))
        self.check()

    def store(self, new_user_matrix: np.ndarray = None, new_item_matrix: np.ndarray = None, new_interaction_matrix: np.ndarray = None):
        if new_user_matrix:
            assert len(self.all_user_matrix.shape) == len(new_user_matrix.shape) and self.all_user_matrix.shape[1] == new_user_matrix.shape[1], f"new_user_matrix {new_user_matrix.shape} not match all_user_matrix {self.all_user_matrix.shape}"
            self.all_user_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)

        if new_item_matrix:
            assert len(self.all_item_matrix.shape) == len(new_item_matrix.shape) and self.all_item_matrix.shape[1] == new_item_matrix.shape[1], f"new_item_matrix {new_item_matrix.shape} not match all_item_matrix {self.all_item_matrix.shape}"
            self.all_item_matrix = np.stack((self.all_user_matrix, new_user_matrix), axis=0)
        
        if new_interaction_matrix:
            assert len(self.all_interaction_matrix.shape) == len(new_interaction_matrix.shape) and self.all_interaction_matrix.shape[1] == new_interaction_matrix.shape[1], f"new_interaction_matrix {new_interaction_matrix.shape} not match all_interaction_matrix {self.all_interaction_matrix.shape}"
            self.all_interaction_matrix = np.stack((self.all_interaction_matrix, new_interaction_matrix), axis=0)
            
            user_ids = self.all_interaction_matrix[:,0].tolist()    # user_ids are stored at 1st col
            self.user_ids = list(set(user_ids))
            self.user_ids.sort()
            
            item_ids = self.all_interaction_matrix[:,1].tolist()    # item_ids are stored at 2nd col
            self.item_ids = list(set(item_ids))
            self.item_ids.sort()

            self.label_ids = {}
            for user_id in self.user_ids:
                mask = (self.all_interaction_matrix[:,0] == user_id) & (self.all_interaction_matrix[:,-1] >= 1)
                self.label_ids[user_id] = list(set(self.all_interaction_matrix[mask][:,1].tolist()))
        
        if new_user_matrix or new_item_matrix or new_interaction_matrix:
            self.check()
    
    def save(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        
        data_path = os.path.join(self.data_path, f"{self.data_name}.pickle")
        with open(data_path, "wb") as f:
            pickle.dump((self.all_user_matrix, self.all_item_matrix, self.all_interaction_matrix), f)

    def set_online_checker(self, online_user_ids: List[int] = None, online_ratio: float = 0.2):
        self.session_id = 0
        self.session2user = {}
        
        if not online_user_ids:
            online_user_ids = random.choices(self.user_ids, k=int(online_ratio*len(self.user_ids)))
        
        self.online_user_ids = online_user_ids

        max_session_id = len(self.online_user_ids)
        return max_session_id
       
    def _compute_seq_data(self):
        all_seq_data, all_seq_len = {}, {}
        all_user_ids = set(self.all_interaction_matrix[:,0].tolist())
        
        for user_id in all_user_ids:
            _user_matrix = self.all_user_matrix[self.all_user_matrix[:,0] == user_id]
            _interaction_matrix = self.all_interaction_matrix[self.all_interaction_matrix[:,0] == user_id]
            _label_matrix = _interaction_matrix[:,-1].tolist()   # last col is label
            _item_ids = _interaction_matrix[:,1].tolist()
            _item_matrices = []
            for _item_id in _item_ids:
                _item_matrix = self.all_item_matrix[self.all_item_matrix[:,0] == _item_id]
                _item_matrix = np.concatenate((_user_matrix, _item_matrix), axis=-1) # concat user_feat and item_feat to make item_feat
                _item_matrices.append(_item_matrix)

            if len(_item_matrices) < self.pad_len:
                len2pad = self.pad_len - len(_item_matrices)
                for _ in range(len2pad):
                    _item_matrix = np.full(shape=_item_matrix.shape, fill_value=-1)
                    _item_matrices.append(_item_matrix)
                _label_matrix.extend([-1 for _ in range(len2pad)])
                all_seq_len.update({user_id: self.pad_len - len2pad})
            elif len(_item_matrices) > self.pad_len:
                _item_matrices = _item_matrices[-self.pad_len:]
                _label_matrix = _label_matrix[-self.pad_len:]
                all_seq_len.update({user_id: self.pad_len})
            else:
                all_seq_len.update({user_id: self.pad_len})

            _item_matrices = np.concatenate(_item_matrices, axis=0)
            _label_matrix = np.array(_label_matrix)[:,np.newaxis]
            _item_matrices = np.concatenate((_item_matrices, _label_matrix), axis=-1)
            all_seq_data.update({user_id: _item_matrices})

        return (all_seq_data, all_seq_len)
    
    def _save_seq_data(self, all_seq_data: Dict, all_seq_len: Dict):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        
        data_path = os.path.join(self.data_path, f"{self.data_name}_seq.pickle")
        with open(data_path, "wb") as f:
            pickle.dump((all_seq_data, all_seq_len), f)
    
    def set_offline_trainer(self, split_ratio: Dict = {"train": (0, 0.8), "valid": (0.8, 1)}, enable_save: bool = False):            
        data_path = os.path.join(self.data_path, f"{self.data_name}_seq.pickle")
        if os.path.exists(data_path) and enable_save:
            with open(data_path, "wb") as f:
                all_seq_data, all_seq_len = pickle.load(f)
        else:
            all_seq_data, all_seq_len = self._compute_seq_data()
            if enable_save:
                self._save_seq_data(all_seq_data=all_seq_data, all_seq_len=all_seq_len)

        dataset = Dataset4Trainer(seq_data=all_seq_data, seq_len=all_seq_len, split_ratio=split_ratio) 
        return dataset

    def set_session(self, trainer: BaseTrainer, device, user_id: int = None, enable_overlap: bool = False, enable_save: bool = False):   # to simulate online checking and offline training
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
        session_mask = np.isin(self.all_item_matrix[:, 0], candidate_item_ids)
        candidate_item_matrix = self.all_item_matrix[session_mask]

        candidate_label_item_ids = list(set(self.label_ids[user_id]) & set(candidate_item_ids))
        candidate_label_attribute_ids = self._compute_label_attribute_ids(data_matrix=candidate_item_matrix, label_ids=candidate_label_item_ids)
    
        if self.score_func == "model":
            data_path = os.path.join(self.data_path, f"{self.data_name}_seq.pickle")
            if os.path.exists(data_path) and enable_save:
                with open(data_path, "wb") as f:
                    all_seq_data, all_seq_len = pickle.load(f)
            else:
                all_seq_data, all_seq_len = self._compute_seq_data()
                if enable_save:
                    self._save_seq_data(all_seq_data=all_seq_data, all_seq_len=all_seq_len)
            
            seq_data, seq_len = all_seq_data[user_id], all_seq_len[user_id]

            new_seq_data = []
            for item_id in range(len(candidate_item_matrix)):
                data = seq_data[:seq_len,:]
                user_matrix = self.all_user_matrix[user_id]
                item_matrix = candidate_item_matrix[item_id]
                label = [-1]
                
                data2predict = np.concatenate((user_matrix, item_matrix, label))
                data2predict = data2predict[np.newaxis,:]
                data = np.concatenate((data, data2predict), axis=0)

                if seq_len < len(seq_data):
                    pad = seq_data[-(len(seq_data)-seq_len):,:]
                    data = np.concatenate((data, pad), axis=0)
                new_seq_data.append(data)

            dataset = Dataset4Checker(seq_data=new_seq_data, seq_len=seq_len)
            dataloader = DataLoader(dataset=dataset)
            
            trainer.to(device)
            trainer.eval()
            score = []
            with torch.no_grad():
                for data in dataloader:
                    x, y = data
                    x = x.to(device)
                    y_ = trainer(x, y.shape[1])
                    score.extend(y_.cpu().tolist())
            score = np.array(score)
            candidate_item_matrix = np.concatenate((candidate_item_matrix, score), axis=1)
        
        elif self.score_func == "embedding":
            raise NotImplementedError
        else:
            raise NotImplementedError
    
        return (candidate_item_matrix, candidate_label_item_ids, candidate_label_attribute_ids)

    def _compute_label_attribute_ids(self, data_matrix: np.ndarray, label_ids: List[int]): 
        label_mask = np.isin(data_matrix[:,0], label_ids)
        label_data_matrix = data_matrix[label_mask]
        attribute_ids = [_ for _ in range(1, label_data_matrix.shape[1])]  # 1st is item_id

        label_attribute_ids = {}
        for attribute_id in attribute_ids:
            attribute_dict = Counter(label_data_matrix[:,attribute_id])
            label_attribute_ids.update({attribute_id: list(attribute_dict.keys())})

        return label_attribute_ids