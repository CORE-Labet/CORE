import numpy as np
import torch
import os

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from typing import List
from collections import Counter
from checker import BaseChecker
from trainer import BaseTrainer
from render import BaseRender

from render import NOT_KNOW_SINGAL, YES_SINGAL, NO_SINGAL
from render import QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL


class ConversationalAgent():
    def __init__(self, checker: BaseChecker, trainer: BaseTrainer, render: BaseRender, cold_start: bool = True):
        self.checker = checker
        self.trainer = trainer
        self.render = render

        self.turn_id = 0
        self.cold_start = cold_start
        self.data_matrix = None
        self.item_ids = []
        self.attribute_ids = {}

        self.checked_item_ids = []  # record checked items
        self.checked_attribute_ids = {} # record checked attributes

    def set_session(self, session_id: int, data_matrix: np.ndarray):
        self.turn_id = 0
        self.session_id = session_id
        self.data_matrix = data_matrix

        if self.cold_start:
            self.data_matrix[:,-1] = 1

        self.item_ids = data_matrix[:,0].astype(int).tolist()
        self.attribute_ids = {} 
        attribute_ids = [_ for _ in range(1, self.data_matrix.shape[1] - 1)]  # 1st is item_id, last col is label
        for attribute_id in attribute_ids:
            attribute_dict = Counter(data_matrix[:,attribute_id])
            attribute_values = list(attribute_dict.keys())
            attribute_values.sort()
            self.attribute_ids.update({attribute_id: attribute_values})
        
        self.checked_item_ids = [] 
        self.checked_attribute_ids = {}
    
    def set_turn(self, query_type: str, query_id, response):
        print("TYPE: ", query_type)
        print("ID: ", query_id)
        print("RESPONSE: ", response)
        self.turn_id += 1
        if query_type == QUERY_ITEM_SIGNAL:
            assert response == NO_SINGAL
            self._update_item(query_item_ids=query_id)
        elif query_type == QUERY_ATTRIBUTE_SINGAL:
            if response == NOT_KNOW_SINGAL:
                self.checked_attribute_ids.update({query_id: self.attribute_ids[query_id]})
                self.attribute_ids.pop(query_id)
            else:
                assert isinstance(response, List)
                self._update_attribute(query_attribute_id=query_id, response_vals=response)
        elif query_type == QUERY_ATTRIBUTE_VAL_SIGNAL:
            assert isinstance(response, List)
            query_attribute_id, query_attribute_vals = query_id
            self._update_attribute_val(query_attribute_id=query_attribute_id, query_attribute_vals=query_attribute_vals, response_vals=response)
        else:
            print(f"query type must be in {[QUERY_ITEM_SIGNAL, QUERY_ATTRIBUTE_SINGAL, QUERY_ATTRIBUTE_VAL_SIGNAL]}")
            raise NotImplementedError
        
        self._sort()    # for convenience to debug

    def _sort(self):
        self.checked_item_ids.sort()
        self.item_ids.sort()
        
        self.checked_attribute_ids = dict(sorted(self.checked_attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.checked_attribute_ids.values():
            attribute_vals.sort()
        self.attribute_ids = dict(sorted(self.attribute_ids.items(), key=lambda x: x[0]))
        for attribute_vals in self.attribute_ids.values():
            attribute_vals.sort()
        
        print("CHECKED ITEM IDS: ", self.checked_item_ids)
        print("CHECKED ATTRIBUTE_IDS: ", self.checked_attribute_ids)
        print("ITEM IDS: ", self.item_ids)
        print("ATTRIBUTE IDS: ", self.attribute_ids)
        
    def _update_item(self, query_item_ids: List[int]):
        self.checked_item_ids.extend(query_item_ids)
        
        for query_item_id in query_item_ids:
            self.item_ids.remove(query_item_id)
    
    def _update_attribute(self, query_attribute_id: int, response_vals: List[int]):
        self.checked_attribute_ids.update({query_attribute_id: self.attribute_ids[query_attribute_id]})
        self.attribute_ids.pop(query_attribute_id) 

        label_mask = np.isin(self.data_matrix[:,query_attribute_id], response_vals)
        label_data_matrix = self.data_matrix[label_mask]
        label_item_ids = label_data_matrix[:,0].astype(int).tolist()
        self.checked_item_ids.extend([idx for idx in self.item_ids if idx not in label_item_ids])
        self.item_ids = list(set(self.item_ids) & set(label_item_ids))
    
    def _update_attribute_val(self, query_attribute_id: int, query_attribute_vals: List[int], response_vals: List[int]):
        if response_vals:   # if receive yes, stop further querying the attribute id
            self.checked_attribute_ids.update({query_attribute_id: self.attribute_ids[query_attribute_id]})
            self.attribute_ids.pop(query_attribute_id) 
            label_mask = np.isin(self.data_matrix[:,query_attribute_id], response_vals)
            label_data_matrix = self.data_matrix[label_mask]
            label_item_ids = label_data_matrix[:,0].astype(int).tolist()
            
            self.checked_item_ids.extend([idx for idx in self.item_ids if idx not in label_item_ids])
            self.item_ids = list(set(self.item_ids) & set(label_item_ids))
        else:
            if query_attribute_id in self.checked_attribute_ids:
                self.checked_attribute_ids[query_attribute_id].extend(query_attribute_vals)
            else:
                self.checked_attribute_ids[query_attribute_id] = query_attribute_vals

            if self.attribute_ids[query_attribute_id] == query_attribute_vals:
                self.attribute_ids.pop(query_attribute_id)
            else:
                self.attribute_ids[query_attribute_id] = [idx for idx in self.attribute_ids[query_attribute_id] if idx not in query_attribute_vals]
            
    def check(self):
        return self.checker.act(data_matrix=self.data_matrix, item_ids=self.item_ids, attribute_ids=self.attribute_ids, turn_id=self.turn_id)  
    
    def evaluate(self):
        query_type, query_id = self.checker.evaluate(data_matrix=self.data_matrix, item_ids=self.item_ids)
        assert query_type is QUERY_ITEM_SIGNAL, f"during evaluation, query type {query_type} must be {QUERY_ITEM_SIGNAL}"
        return query_id
    
    def load(self, args):
        self.trainer.load_state_dict(torch.load(args.save_dir, map_location=args.device))
    
    @staticmethod
    def _evaluate_with_criterion(y_, y, criterion, threshold: float = 0.5):
        if y.dtype != torch.float32:
            y = y.to(torch.float32)
        loss = criterion(y_, y)
        acc = torch.mean(torch.eq(y_ >= threshold, y), dtype=torch.float).item()
        y_, y = [_.view(-1).detach().cpu().numpy() for _ in (y_, y)]
        return (loss, acc, roc_auc_score(y_true=y, y_score=y_))
    
    def _train_trainer(self, args, dataloader, optimizer, scheduler, criterion):
        self.trainer.train()

        for data in tqdm(dataloader):
            data = [_.to(args.device, non_blocking=True) for _ in data]
            x, y = data
            y_ = self.trainer(x, y.shape[1])
            loss, acc, auc = self._evaluate_with_criterion(y_=y_, y=y, criterion=criterion)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainer.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            scheduler.step(auc)

    def _evaluate_trainer(self, args, dataloader, criterion):
        self.trainer.eval()
        res = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                x, y = data
                x = x.to(args.device)
                y_ = self.trainer(x, y.shape[1])
                res.append((y_.cpu(), y))
        res = [torch.cat(_) for _ in list(zip(*res))]
        loss, acc, auc = self._evaluate_with_criterion(*res, criterion=criterion)
        return (loss, acc, auc)

    def train(self, args, dataset: Dataset):
        self.trainer.to(args.device)

        optimizer = torch.optim.AdamW(self.trainer.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.9, 
                            patience=len(dataset)//(args.batch_size*5), verbose=True, min_lr=args.min_lr)
        dataloader = DataLoader(dataset=dataset)

        if not args.save_path:
            current_path = os.path.abspath(os.getcwd())
            save_path = os.path.join(os.path.dirname(current_path), "log/")
        else:
            save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True) 
        save_path = os.path.join(save_path, f"{args.trainer}.pt")
        
        best_auc = 0
        for _ in range(args.epochs):
            dataset.set_mode("train")
            self._train_trainer(args=args, dataloader=dataloader, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
            dataset.set_mode("valid")
            res = self._evaluate_trainer(args=args, dataloader=dataloader, criterion=criterion)
            auc = res[-1]
            if auc > best_auc:
                best_auc = auc
                torch.save(self.trainer.state_dict(), save_path)
        return best_auc

    def check_with_render(self, response: str) -> str:
        assert isinstance(self.render, BaseRender)
        item_ids, attribute_ids = self.render.response2ids(response=response)
        self.set_turn(item_ids=item_ids, attribute_ids=attribute_ids)
        _, _, query_type, query_id = self.check()
        raise self.render.ids2query(query_type=query_type, query_id=query_id)