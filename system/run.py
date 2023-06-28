import argparse
import torch

from typing import List
from evaluator import evaluate_online_checker

def run(args):
    average_turn, average_success_rate = evaluate_online_checker(args)
    print(f"===== AVG TRUN: {average_turn}, AVG SR: {average_success_rate} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument()
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--cuda", help="gpu device", default=0, type=int)
    parser.add_argument("--dataset", help="name of dataset", choices=["taobao", "tmall", "alipay", "amazon", "movielen"])

    parser.add_argument("--retriever", help="name of retriever", default="random", choices=["time", "random"])
    parser.add_argument("--candidate_items", help="number of candidate items per session", default=30, type=int)
    parser.add_argument("--pos_neg_ratio", help="ratio of positive and negative items in data sampling", default=0.1, type=float)
    parser.add_argument("--render", help="name of render", default="rule", choices=["rule", "lm"])
    parser.add_argument("--enable_quit", help="allow user agent to quit the session when the number of turns reach max_turn", action="store_true")
    parser.add_argument("--max_turn", help="max number of turns for user agent", default=15, type=int)
    parser.add_argument("--enable_not_know", help="allow user agent to response with NOT_KNOW_SINGAL", action="store_true")
    parser.add_argument("--num_not_know", help="number of preferred attribute values to answer with NOT_KNOW_SINGAL", default=3, type=int)
    
    parser.add_argument("--checker", help="name of checker", default="core", choices=["core", "item", "attribute"])
    parser.add_argument("--n_items", help="top n_items at each turn", default=1, type=int)
    parser.add_argument("--n_attribute_val", help="top n_attribute_val at each turn", default=1, type=int)
    parser.add_argument("--query_attribute_val", help="query attribute values instead of querying attribute ids", action="store_false")
    parser.add_argument("--query_attribute_only", help="let core checker only query attribute ids or attribute values", action="store_true")
    parser.add_argument("--query_item_only", help="let core checker only query items", action="store_true")
    parser.add_argument("--enable_penalty", help="enable to push the core checker to recommend items", action="store_true")
    parser.add_argument("--penalty_weight", help="weight for the penalty", default=0.0, type=float)
    parser.add_argument("--cold_start", help="let the recommender system cold-start", action="store_true")

    parser.add_argument("--trainer", help="name of trainer", default="fm", choices=["fm", "deepfm", "pnn", "esmm", "esmm2", "mmoe", "lstm", "gru", "din"])
    parser.add_argument("--input_size", help="size of input layer", default=48, type=int)
    parser.add_argument("--hidden_size", help="size of hidden layers", default=64, type=int)
    parser.add_argument("--dropout", help="dropout", default=0.5, type=float)
    parser.add_argument("--pre_hidden_sizes", help="size of hidden layers", default=[256, 64, 16, 1], type=List[int])
    parser.add_argument("--epochs", help="number of training epochs", default=8, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-2, type=float)
    parser.add_argument("--min_lr", help="min learning rate to clip", default=5e-4, type=float)
    parser.add_argument("--l2_reg", help="weight for l2 regularization", default=1e-4, type=float)
    parser.add_argument("--batch_size", help="batch size for training", default=128, type=int)

    args = parser.parse_args()
    device = "cpu" if args.cuda < 0 else f"cuda:{args.cuda}"
    args.device = torch.device(device)