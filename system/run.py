import argparse
import torch
import os
import pickle
import numpy as np

from typing import List, Dict
from evaluator import load_agent, evaluate_online_checker, evaluate_offline_trainer, evaluate_offline_online_loop

# data is organized into three np.ndarry to store:
# user_matrix: 1st col: user_ids
# item_matrix: 1st col: item_ids
# interaction_matrix: 1st user_ids, 2nd item_ids, 3rd time_stamp, 4th labels
def run_data_dealer(args, redo=False):
    current_path = os.path.abspath(os.getcwd()) 
    data_path = os.path.join(os.path.dirname(current_path), "data/")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if not os.path.exists(os.path.join(data_path, args.dataset) + ".pickle") or redo:
        user_ids = np.arange(0, 10)
        user_matrix = np.random.randint(low=0, high=5, size=(10, 9))
        user_matrix = np.column_stack((user_ids, user_matrix))
        item_ids = np.arange(0, 50)
        item_matrix = np.random.randint(low=0, high=5, size=(50, 9))
        item_matrix = np.column_stack((item_ids, item_matrix))
        
        scores = np.random.randint(low=0, high=2, size=100)
        scores[0] = 1   # at least one item can satisfy the user demand

        times = np.random.randint(low=0, high=2, size = 100)
        indices = np.random.choice(len(user_ids) * len(item_ids), size=100, replace=False)
        rows = indices // len(item_ids)
        cols = indices % len(item_ids)
        interaction_matrix = np.column_stack((user_ids[rows], item_ids[cols]))
        interaction_matrix = np.column_stack((interaction_matrix, times, scores))

        with open(os.path.join(data_path, f"{args.dataset}.pickle"), "wb") as f:
            pickle.dump((user_matrix, item_matrix, interaction_matrix), f)
    

def run(args):
    manager, conversational_agent, user_agent = load_agent(args)
    if args.use_loop:
        evaluate_offline_online_loop(args, manager, conversational_agent)
    else:
        if args.use_tune:
            best_auc, trainer = evaluate_offline_trainer(args, manager, conversational_agent)
            print(f"===== BEST AUC: {best_auc} =====")
            conversational_agent.trainer = trainer
        average_turn, average_success_rate = evaluate_online_checker(args, manager, conversational_agent, user_agent)
        print(f"===== AVG TRUN: {average_turn}, AVG SR: {average_success_rate} =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--cuda", help="gpu device", default=0, type=int)
    parser.add_argument("--dataset", help="name of dataset", default="taobao", choices=["taobao", "tmall", "alipay", "amazon", "movielen"])
    parser.add_argument("--split_ratio", help="split ratio of splitting dataset into offline_train, offline_validation datasets", default={"train": (0, 0.8), "valid": (0.8, 1)}, type=Dict)
    parser.add_argument("--online_ratio", help="split ratio of splitting user_ids into online_user_ids", default=0.2, type=float)
    parser.add_argument("--num_session", help="number of sessions", default=4, type=int)
    parser.add_argument("--num_turn", help="number of turns per session", default=4, type=int)
    parser.add_argument("--failure_penalty", help="penlty of number of turns for each session", default=3, type=int)

    parser.add_argument("--num_loop", help="number of loops in loop setting", default=2, type=int)
    parser.add_argument("--use_loop", help="use offline-online loop or not", action="store_true")
    
    parser.add_argument("--use_tune", help="tune trainer or not", action="store_true")
    parser.add_argument("--retriever", help="name of retriever", default="random", choices=["time", "random"])
    parser.add_argument("--candidate_items", help="number of candidate items per session", default=30, type=int)
    parser.add_argument("--pos_neg_ratio", help="ratio of positive and negative items in data sampling", default=0.1, type=float)
    parser.add_argument("--render", help="name of render", default="rule", choices=["rule", "lm"])
    parser.add_argument("--enable_quit", help="allow user agent to quit the session when the number of turns reach max_turn", action="store_true")
    parser.add_argument("--max_turn", help="max number of turns for user agent", default=15, type=int)
    parser.add_argument("--enable_not_know", help="allow user agent to response with NOT_KNOW_SINGAL", action="store_true")
    parser.add_argument("--num_not_know", help="number of preferred attribute values to answer with NOT_KNOW_SINGAL", default=2, type=int)
    
    parser.add_argument("--checker", help="name of checker", default="core", choices=["core", "item", "attribute"])
    parser.add_argument("--n_items", help="top n_items at each turn", default=1, type=int)
    parser.add_argument("--n_attribute_val", help="top n_attribute_val at each turn", default=1, type=int)
    parser.add_argument("--query_attribute_val", help="query attribute values instead of querying attribute ids", action="store_true")
    parser.add_argument("--query_attribute_only", help="let core checker only query attribute ids or attribute values", action="store_true")
    parser.add_argument("--query_item_only", help="let core checker only query items", action="store_true")
    parser.add_argument("--enable_penalty", help="enable to push the core checker to recommend items", action="store_true")
    parser.add_argument("--penalty_weight", help="weight for the penalty", default=0.0, type=float)
    parser.add_argument("--cold_start", help="let the recommender system cold-start", action="store_true")

    parser.add_argument("--trainer", help="name of trainer", default="esmm", choices=["fm", "deepfm", "pnn", "esmm", "esmm2", "mmoe", "lstm", "gru", "din"])
    parser.add_argument("--input_size", help="size of input layer", default=48, type=int)
    parser.add_argument("--hidden_size", help="size of hidden layers", default=64, type=int)
    parser.add_argument("--dropout", help="dropout", default=0.5, type=float)
    parser.add_argument("--pre_hidden_sizes", help="size of hidden layers", default=[256, 64, 16, 1], type=List[int])
    parser.add_argument("--epochs", help="number of training epochs", default=8, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-2, type=float)
    parser.add_argument("--min_lr", help="min learning rate to clip", default=5e-4, type=float)
    parser.add_argument("--l2_reg", help="weight for l2 regularization", default=1e-4, type=float)
    parser.add_argument("--batch_size", help="batch size for training", default=4, type=int)
    parser.add_argument("--pad_len", help="padding len of each sequence of user", default=10, type=int)
    parser.add_argument("--num_workers", help="number of workers for dataloader", default=1, type=int)
    parser.add_argument("--save_path", help="path to save trainer", default="", type=str)
    parser.add_argument("--score_func", help="method to generate prediction scores", choices=["embedding", "model"], default="model", type=str)

    args = parser.parse_args()
    device = "cpu" if args.cuda < 0 else f"cuda:{args.cuda}"
    args.device = torch.device(device)
 
    run_data_dealer(args=args, redo=True)  # generate sampled data
    run(args=args)
