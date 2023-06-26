import numpy as np
import json
import os
import pandas as pd

from collections import defaultdict


def load_openai_key(key_file: str = None):
    if not key_file:
        current_path = os.path.abspath(os.getcwd())
        key_file = os.path.join(os.path.dirname(current_path), "openai_key.json") 
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            openai_key = json.load(f)
        assert isinstance(openai_key, str), f"openai_key {type(openai_key)} must be str"
        return openai_key
    else:
        raise FileNotFoundError


class DataDealer():
    def __init__(self, data_name: str, min_len: int = 5) :
        assert data_name in ["taobao", "tmall", "alipay", "amazon", "movielen"], f"data_name must be either taobao, tmall, alipay, amazon, or movielen"
        self.min_len = min_len

        current_path = os.path.abspath(os.getcwd())
        self.data_path = {
            "taobao": os.path.join(os.path.dirname(current_path), "taobao/raw_data"),
            "tmall": os.path.join(os.path.dirname(current_path), "tmall/raw_data"),
            "alipay": os.path.join(os.path.dirname(current_path), "alipay/raw_data"),
            "amazon": os.path.join(os.path.dirname(current_path), "amazon/raw_data"),
            "movielen": os.path.join(os.path.dirname(current_path), "movielen/raw_data")
        }[data_name]
        self.data_func = {
            "taobao": self._deal_taobao,
            "tmall": self._deal_tmall,
            "alipay": self._deal_alipay,
            "amazon": self._deal_amazon,
            "movielen": self._deal_movielen
        }[data_name]

        self.data_func()

    def _deal_taobao(self):
        csv_file = pd.read_csv(os.path.join(self.data_path, 'UserBehavior.csv'), header=None)
        # sort data by user and timestamp
        csv_file = csv_file.sort_values(by=[0, 4])
        # put label in the last col, timestamp at the second last col, category at the thrid last col
        csv_file = csv_file[[0, 1, 2, 4, 3]]
        csv_file.rename(columns={4: 3, 3: 4}, inplace=True)
        csv_file[4].replace(['pv', 'cart', 'fav', 'buy'], [0, 1, 1, 1], inplace=True)

        np_file = csv_file.to_numpy()
        # split by user
        is_begin = np.append([1], np.diff(np_file[:, 0])) != 0
        # np_file[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(np_file[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [np_file.shape[0]]))
        # filter out behaviors shorter than min_len 
        np_file = np_file[behavior_len[belong_inter] >= self.min_len]
        behavior_len = behavior_len[behavior_len >= self.min_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])

        num_field = np.shape(np_file)[1] - 2
        num_field_feat = np.array([0] * num_field)  # record feature_num in each field
        # remap field
        for field_id in range(num_field):
            remap_index = np.unique(np_file[:, field_id], return_inverse=True)[1]
            num_field_feat[field_id] = np.max(remap_index) + 1
            np_file[:, field_id] = remap_index + np.sum(num_field_feat[:field_id])

        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=np_file,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=num_field_feat
                 )
    
    def _deal_tmall(self):
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'user_log_format1.csv'))
        user_info_csv = pd.read_csv(os.path.join(self.data_path, 'user_info_format1.csv'))
        # sort data by user and timestamp
        user_log_csv = user_log_csv.sort_values(by=['user_id', 'time_stamp'])
        # sort user data by user id
        user_info_csv = user_info_csv.sort_values(by='user_id')
        # put label in the last col, timestamp at the second last col, category at the thrid last col
        user_log_csv = user_log_csv[
            ['user_id', 'item_id', 'seller_id', 'brand_id', 'cat_id', 'time_stamp', 'action_type']]
        user_info_np = user_info_csv.to_numpy().astype(np.int32)
        user_log_np = user_log_csv.to_numpy().astype(np.int32)
        user_log_np = user_log_np[np.sort(np.unique(user_log_np, axis=0, return_index=True)[1])]
        user_log_np[:, -1][user_log_np[:, -1] >= 1] = 1

        month = user_log_np[:, -2] // 100
        day = user_log_np[:, -2] % 100
        days = np.append([0], np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        user_log_np[:, -2] = days[month - 1] + day
        
        # split by user
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))
        # filter out behaviors shorter than min_len 
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.min_len]
        user_info_np = user_info_np[:len(behavior_len)][behavior_len >= self.min_len]
        behavior_len = behavior_len[behavior_len >= self.min_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        num_field = np.shape(user_log_np)[1] - 2
        num_field_feat = np.array([0] * num_field)
        # remap field
        for field_id in range(num_field):
            remap_index = np.unique(user_log_np[:, field_id], return_inverse=True)[1]
            num_field_feat[field_id] = np.max(remap_index) + 1
            user_log_np[:, field_id] = remap_index + np.sum(num_field_feat[:field_id])
            if field_id == num_field - 1:
                user_log_np[:, field_id] = remap_index

        user_info_np = user_info_np[:, 1:]
        for field_id in range(2):
            remap_index = np.unique(user_info_np[:, field_id], return_inverse=True)[1]
            num_field_feat = np.append(num_field_feat, np.max(remap_index) + 1)
            user_info_np[:, field_id] = remap_index

        user_log_np = np.concatenate(
            [user_log_np[:, :-3], user_info_np[user_log_np[:, 0]], user_log_np[:, -3:]],
            axis=-1)

        num_field_feat[[-3, -2, -1]] = num_field_feat[[-2, -1, -3]]
        for field in [-3, -2, -1]:
            user_log_np[:, field - 2] += np.sum(num_field_feat[:field])
        
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=num_field_feat
                 )

    def _deal_alipay(self):
        csv_file = pd.read_csv(os.path.join(self.data_path, 'ijcai2016_taobao.csv'))
        # sort data by user and timestamp
        csv_file = csv_file.sort_values(by=['use_ID', 'time'])
        # put label in the last col, timestamp at the second last col, category at the thrid last col
        csv_file = csv_file[['use_ID', 'sel_ID', 'ite_ID', 'cat_ID', 'time', 'act_ID']]
        np_file = csv_file.to_numpy()
        np_file = np_file[np.sort(np.unique(np_file, axis=0, return_index=True)[1])]
        
        month_day = np_file[:, -2] % 10000
        month = month_day // 100
        day = month_day % 100
        days = np.append([0], np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        np_file[:, -2] = days[month - 1] + day
        # split by user
        is_begin = np.append([1], np.diff(np_file[:, 0])) != 0
        # np_file[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(np_file[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [np_file.shape[0]]))
        # filter out behaviors shorter than min_len 
        np_file = np_file[behavior_len[belong_inter] >= self.min_len]
        behavior_len = behavior_len[behavior_len >= self.min_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        
        num_field = np.shape(np_file)[1] - 2
        num_field_feat = np.array([0] * num_field)  # record feature_num in each field
        # remap field
        for field_id in range(num_field):
            remap_index = np.unique(np_file[:, field_id], return_inverse=True)[1]
            num_field_feat[field_id] = np.max(remap_index) + 1
            np_file[:, field_id] = remap_index + np.sum(num_field_feat[:field_id])
            # if field_id == num_field - 1:
            #     cate_count = np.bincount(remap_index)
            #     print(cate_count.min(), cate_count.max())

        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=np_file,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=num_field_feat
                 )

    def _deal_amazon2csv(self):
        user_log_json = os.path.join(self.data_path, "Electronics.json")
        item_info_json = os.path.join(self.data_path, "meta_Electronics.json")
        useful_log_field = ['reviewerID', 'asin', 'style', 'vote', 'unixReviewTime', 'overall']
        useful_item_field = ['asin', 'price', 'main_cat', 'brand', 'category']
        log_csv = defaultdict(lambda: {})
        item_csv = defaultdict(lambda: {})
        count = 0
        with open(user_log_json) as user_log:
            for user_log_line in user_log:
                user_log_line = json.loads(user_log_line)
                for field in user_log_line.keys():
                    if field in useful_log_field:
                        log_csv[i][field] = user_log_line[field]
                count += 1

        log_csv = pd.DataFrame.from_dict(log_csv, orient='index')[useful_log_field]
        log_csv.to_csv(os.path.join(self.data_path, 'user_log.csv'), index=False)

        count = 0
        with open(item_info_json) as item_info:
            for item_info_line in item_info:
                item_info_line = json.loads(item_info_line)
                for field in item_info_line.keys():
                    if field in useful_item_field:
                        if field == 'category':
                            item_info_line[field].remove('Electronics')
                        item_csv[i][field] = item_info_line[field]
                i += 1

        item_csv = pd.DataFrame.from_dict(item_csv, orient='index')[useful_item_field]
        item_csv.to_csv(os.path.join(self.data_path, 'item_info.csv'), index=False)

        del log_csv
        del item_csv
  
    def _deal_amazon(self):
        if not os.path.exists(os.path.join(self.data_path, "user_log.csv")):
            self._deal_amazon2csv()
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'user_log.csv'))
        user_log_csv = user_log_csv[['reviewerID', 'asin', 'vote', 'unixReviewTime', 'overall']]
        user_log_csv['vote'].replace(np.nan, 0, inplace=True)
        user_log_csv = user_log_csv.astype(str)
        user_log_csv.sort_values(by=['reviewerID', 'unixReviewTime'], inplace=True)

        item_info_csv = pd.read_csv(os.path.join(self.data_path, 'item_info.csv'))
        # main_category = item_info_csv['main_cat'].to_numpy().astype(str)
        item_info_csv = item_info_csv[['asin', 'price', 'brand', 'main_cat', 'category']]
        item_info_csv['price'].replace(np.nan, '$0', inplace=True)

        user_log_np = user_log_csv.to_numpy()
        item_info_np = item_info_csv.to_numpy()

        item_info_np = item_info_np[np.unique(item_info_np[:, 0].astype(str), return_index=True)[1]]
        user_log_np = user_log_np[np.isin(user_log_np[:, 1].astype(str), item_info_np[:, 0].astype(str))]
        item_info_np = item_info_np[np.isin(item_info_np[:, 0].astype(str), user_log_np[:, 1].astype(str))]
        # sort item data by item_id
        item_info_np = item_info_np[np.argsort(item_info_np[:, 0].astype(str)), 1:]
        # remap item_id
        user_log_np[:, 1] = np.unique(user_log_np[:, 1].astype(str), return_inverse=True)[1]
        # remap user_id
        user_log_np[:, 0] = np.unique(user_log_np[:, 0].astype(str), return_inverse=True)[1]
        user_log_np[:, 2] = np.char.replace(user_log_np[:, 2].astype(str), ',', '')
        user_log_np[:, -1] = np.where(user_log_np[:, -1].astype(float) >= 5, 1, 0)
        user_log_np = user_log_np.astype(float).astype(int)
        # split data by user
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))

        # filter out behaviors shorter than min_len 
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.min_len]
        behavior_len = behavior_len[behavior_len >= self.min_len]

        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        id_exist, item_id_remap = np.unique(user_log_np[:, 1], return_inverse=True)
        user_log_np[:, 1] = item_id_remap
        item_info_np = item_info_np[id_exist]

        def eval_and_len(i):
            list_i = eval(item_info_np[i, -1])
            list_i.append(item_info_np[i, -2])
            return [list_i, len(list_i)]

        eval_len = np.array([eval_and_len(i) for i in range(item_info_np.shape[0])], dtype=object)
        item_info_np[:, -1] = eval_len[:, 0]
        eval_len = eval_len[:, 1]
        eval_squeeze = np.concatenate(item_info_np[:, -1])
        eval_squeeze = np.unique(eval_squeeze, return_inverse=True)[1]

        # remap field
        item_info_np = item_info_np[:, 0:2]
        item_info_np[:, 0] = np.char.replace(item_info_np[:, 0].astype(str), '$', '')
        item_info_np[:, 0] = np.char.replace(item_info_np[:, 0].astype(str), ',', '')
        item_info_np[:, 0][
            np.logical_not(np.char.isnumeric(np.char.replace(item_info_np[:, 0].astype(str), '.', '')))] = 0
        item_info_np[:, 1] = np.unique(item_info_np[:, 1].astype(str), return_inverse=True)[1]
        item_info_np = item_info_np.astype(float)

        continuous_feature = np.stack([item_info_np[item_id_remap, 0], user_log_np[:, 2]], axis=-1)

        user_log_np[:, 2] = item_info_np[item_id_remap, 1]
        user_log_np[:, 0] = np.unique(user_log_np[:, 0], return_inverse=True)[1]
        # remap field
        num_field = np.shape(user_log_np)[1] - 2 + 1
        num_field_feat = np.array([0] * num_field)  
        for field_id in range(num_field - 1):
            num_field_feat[field_id] = np.max(user_log_np[:, field_id]) + 1
            user_log_np[:, field_id] = user_log_np[:, field_id] + np.sum(num_field_feat[:field_id])
        num_field_feat[-1] = np.max(eval_squeeze) + 1

        eval_squeeze = np.array(np.split(eval_squeeze + np.sum(num_field_feat[:-1]), np.cumsum(eval_len)[:-1]), dtype=np.ndarray)
        user_log_np = np.concatenate([user_log_np, np.expand_dims(eval_squeeze[item_id_remap], axis=-1)], axis=-1)
        user_log_np = user_log_np[:, [0, 1, 2, 5, 3, 4]]
        
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=num_field_feat,
                 continuous_feature=continuous_feature
                 )
    
    def _deal_movielen(self):
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
        item_info_csv = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))

        user_log_csv = user_log_csv[['userId', 'movieId', 'timestamp', 'rating']]
        user_log_csv.sort_values(by=['userId', 'timestamp'], inplace=True)

        user_log_np = user_log_csv.to_numpy()
        item_info_np = item_info_csv['genres'].to_numpy().astype(str)
        user_log_np[:, -1] = np.where(user_log_np[:, -1] >= 4.5, 1, 0)
        user_log_np = user_log_np.astype(int)
        item_info_np = np.char.split(item_info_np, '|')
        split_len = [len(i) for i in item_info_np]
        item_info_np = np.unique(np.concatenate(item_info_np), return_inverse=True)[1]

        # split by user
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))
        # filter out behaviors shorter than min_len 
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.min_len]
        behavior_len = behavior_len[behavior_len >= self.min_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])

        user_log_np[:, 0] = np.unique(user_log_np[:, 0], return_inverse=True)[1]
        user_log_np[:, 1] = np.unique(user_log_np[:, 1], return_inverse=True)[1]

        # remap field
        num_field = np.shape(user_log_np)[1] - 2 + 1
        num_field_feat = np.array([0] * num_field) 
        for field_id in range(num_field - 1):
            num_field_feat[field_id] = np.max(user_log_np[:, field_id]) + 1
            user_log_np[:, field_id] = user_log_np[:, field_id] + np.sum(num_field_feat[:field_id])
        num_field_feat[-1] = np.max(item_info_np) + 1
        item_info_np = np.array(np.split(item_info_np + np.sum(num_field_feat[:-1]), np.cumsum(split_len)[:-1]),
                                dtype=np.ndarray)
        user_log_np = np.concatenate(
            [user_log_np, np.expand_dims(item_info_np[user_log_np[:, 1] - num_field_feat[0]], axis=-1)], axis=-1)
        user_log_np = user_log_np[:, [0, 1, 4, 2, 3]]

        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=num_field_feat,
                 )