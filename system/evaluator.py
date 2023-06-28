import numpy as np
import torch
import random
import os
import logging
import pickle

from user import UserAgent
from agent import ConversationalAgent
from data import DataManager
from trainer import TowerTrainer, SequenceTrainer
from retriever import TimeRetriever, RandomRetriever
from checker import ItemChecker, AttributeChecker, CoreChecker
from render import RuleRender, LMRender

from user import QUIT_SINGAL, YES_SINGAL

logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_render(args):
    if args.render == "rule":
        render = RuleRender()
    elif args.render == "lm":
        render = LMRender()
    else:
        raise NotImplementedError
    return render

def load_retriever(args):
    if args.retriever == "time":
        retriever = TimeRetriever(
            num_candidate_items=args.candidate_items,
            pos_neg_ratio=args.pos_neg_ratio
        )
    elif args.retriever == "random":
        retriever = RandomRetriever(
            num_candidate_items=args.candidate_items,
            pos_neg_ratio=args.pos_neg_ratio
        )
    else:
        raise NotImplementedError
    return retriever


def load_checker(args):
    if args.checker == "item":
        checker = ItemChecker(
            n_items=args.n_items
        )
    elif args.checker == "attribute":
        checker = AttributeChecker(
            n_items=args.n_items,
            n_attribute_val=args.n_attribute_val,
            query_attribute_val=args.query_attribute_val
        )
    elif args.checker == "core":
        checker = CoreChecker(
            n_items=args.n_items,
            n_attribute_val=args.n_attribute_val,
            query_attribute_val=args.query_attribute_val,
            query_attribute_only=args.query_attribute_only,
            query_item_only=args.query_item_only,
            enable_penalty=args.enable_penalty,
            penalty_weight=args.penalty_weight
        )
    else:
        raise NotImplementedError
    return checker


def load_trainer(args, num_feat):
    if args.trainer in ["fm", "deepfm", "pnn", "esmm", "esmm2", "mmoe"]:
        trainer = TowerTrainer(
            num_feat=num_feat,
            input_size=args.input_size,
            hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            model_name=args.trainer
        )
    elif args.trainer in ["lstm", "gru", "din"]:
        trainer = SequenceTrainer(
            num_feat=num_feat,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            model_name=args.trainer
        )
    else:
        raise NotImplementedError
    return trainer


def evaluate_offline_trainer(args):
    pass

def evaluate_online_checker(args):
    set_seed(args.seed)
    retriever = load_retriever(args)
    manager = DataManager(retriever=retriever)
    current_path = os.path.abspath(os.getcwd()) 

    data_path = os.path.join(current_path, "data/")
    data_name = args.dataset + "_" + args.trainer
    with open(os.path.join(data_path, data_name) + ".pickle", "rb") as f:
        user_matrix, item_matrix, interaction_matrix = pickle.load(f)
    manager.load(user_matrix=user_matrix, item_matrix=item_matrix, interaction_matrix=interaction_matrix)
    print("====== LOAD DATA =====")
    max_session_id = manager.set_online_checker(split_ratio=args.split_ratio)
    num_session = min(max_session_id, args.num_session)

    checker = load_checker(args)
    render = load_render(args)
    trainer = load_trainer(args)
    conversational_agent = ConversationalAgent(
        checker=checker, trainer=trainer, render=render, cold_start=args.cold_start
    )
    user_agent = UserAgent(
        max_turn=args.max_turn, enable_not_know=args.enable_not_know, enable_quit=args.enable_quit,
        num_not_know=args.num_not_know, render=render
    )

    num_turn, success_rate = [], []
    failure_turn = args.num_turn + args.failure_penalty
    for session_id in range(num_session):
        print(f"====== SESSION {session_id} =====")
        data_matrix, label_item_ids, label_attribute_ids = manager.set_session()
        conversational_agent.set_session(session_id=session_id, data_matrix=data_matrix)
        user_agent.set_session(session_id=session_id, label_item_ids=label_item_ids, label_attribute_ids=label_attribute_ids)
        
        is_stop =  False
        for turn_id in range(args.num_turn):
            if is_stop:
                break
            
            query_type, query_id = conversational_agent.check()
            response = user_agent.response(query_type=query_type, query_id=query_id)
            if response is YES_SINGAL:
                num_turn.append(turn_id)
                success_rate.append(1)
                is_stop = True
            elif response is QUIT_SINGAL:
                num_turn.append(failure_turn)
                success_rate.append(0)
                is_stop = True
            else:
                conversational_agent.set_turn(query_type=query_type, query_id=query_id, response=response)
        
        if not is_stop:
            query_id = conversational_agent.evaluate()
            response = user_agent.evaluate(query_id)
            if response is YES_SINGAL:
                num_turn.append(args.num_turn)
                success_rate.append(1)
            else:
                num_turn.append(failure_turn)
                success_rate.append(0)
    
    return np.mean(num_turn), np.mean(success_rate)