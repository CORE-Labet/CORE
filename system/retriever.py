import numpy as np
import random

from typing import List 


class BaseRetriever:
    def __init__(self, num_candidate_items: int, pos_neg_ratio: float):
        self.num_candidate_items = num_candidate_items
        self.pos_neg_ratio = pos_neg_ratio
    
    def sample(self, item_ids: List[int]) -> List[int]:
        raise NotImplementedError

    def sample_with_ratio(self, pos_item_ids: List[int], neg_item_ids: List[int]) -> List[int]:
        raise NotImplementedError


class TimeRetriever(BaseRetriever):
    def __init__(self, num_candidate_items: int = 30, pos_neg_ratio: float = 0.1):
        super().__init__(num_candidate_items=num_candidate_items, pos_neg_ratio=pos_neg_ratio)
    
    def sample(self, item_ids: List[int]) -> List[int]:
        assert len(item_ids) >= self.num_candidate_items
        return item_ids[-self.num_candidate_items:] # suppose that item_ids are sorted from past to recent

    def sample_with_ratio(self, pos_item_ids: List[int], neg_item_ids: List[int]) -> List[int]:
        assert len(pos_item_ids) + len(neg_item_ids) >= self.num_candidate_items 
        num_pos = self.num_candidate_items * self.pos_neg_ratio
        num_neg = self.num_candidate_items - num_pos
        if len(pos_item_ids) > num_pos and len(neg_item_ids) > num_neg:
            return pos_item_ids[-num_pos:] + neg_item_ids[-num_neg:]
        if len(pos_item_ids) < num_pos:
            num_neg = self.num_candidate_items - len(pos_item_ids)
            return pos_item_ids + neg_item_ids[-num_neg:]
        if len(neg_item_ids) < num_neg:
            num_pos = self.num_candidate_items - len(neg_item_ids)
            return neg_item_ids + pos_item_ids[-num_pos:]


class RandomRetriever(BaseRetriever):
    def __init__(self, num_candidate_items: int = 30, pos_neg_ratio: float = 0.1):
        super().__init__(num_candidate_items=num_candidate_items, pos_neg_ratio=pos_neg_ratio)
    
    def sample(self, item_ids: List[int]) -> List[int]:
        assert len(item_ids) >= self.num_candidate_items
        return random.sample(item_ids, self.num_candidate_items)

    def sample_with_ratio(self, pos_item_ids: List[int], neg_item_ids: List[int]) -> List[int]:
        assert len(pos_item_ids) + len(neg_item_ids) >= self.num_candidate_items 
        num_pos = self.num_candidate_items * self.pos_neg_ratio
        num_neg = self.num_candidate_items - num_pos
        if len(pos_item_ids) > num_pos and len(neg_item_ids) > num_neg:
            return random.sample(pos_item_ids, num_pos) + random.sample(neg_item_ids, num_neg)
        if len(pos_item_ids) < num_pos:
            return pos_item_ids + random.sample(neg_item_ids, self.num_candidate_items-len(pos_item_ids))
        if len(neg_item_ids) < num_neg:
            return neg_item_ids + random.sample(pos_item_ids, self.num_candidate_items-len(neg_item_ids))