import numpy as np
import torch
import random
import os
import pickle

from user import UserAgent
from agent import ConversationalAgent
from manager import DataManager
from trainer import TowerTrainer, SequenceTrainer
from retriever import TimeRetriever, RandomRetriever
from checker import ItemChecker, AttributeChecker, CoreChecker
from render import RuleRender, LMRender

from render import QUIT_SINGAL, YES_SINGAL, QUERY_ITEM_SIGNAL


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

def load_trainer(args, num_val, num_feat):
    if args.trainer in ["fm", "deepfm", "pnn", "esmm", "esmm2", "mmoe"]:
        trainer = TowerTrainer(
            num_val=num_val,
            num_feat=num_feat,
            input_size=args.input_size,
            hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            model_name=args.trainer
        )
    elif args.trainer in ["lstm", "gru", "din"]:
        trainer = SequenceTrainer(
            num_val=num_val,
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

def load_agent(args):
    set_seed(args.seed)
    retriever = load_retriever(args)
    manager = DataManager(
        data_name=args.dataset, retriever=retriever, pad_len=args.pad_len, score_func=args.score_func
    )
    manager.load()
    print("====== LOAD DATA =====")
    num_val, num_feat = manager.get_statics()

    checker = load_checker(args)
    render = load_render(args)
    trainer = load_trainer(args, num_val=num_val, num_feat=num_feat)
    conversational_agent = ConversationalAgent(
        checker=checker, trainer=trainer, render=render, cold_start=args.cold_start
    )
    user_agent = UserAgent(
        max_turn=args.max_turn, enable_not_know=args.enable_not_know, enable_quit=args.enable_quit,
        num_not_know=args.num_not_know, render=render
    )
    return (manager, conversational_agent, user_agent)


def evaluate_offline_trainer(args, manager: DataManager, conversational_agent: ConversationalAgent):
    dataset = manager.set_offline_trainer(split_ratio=args.split_ratio)
    best_auc = conversational_agent.train(args=args, dataset=dataset)
    return (best_auc, conversational_agent.trainer)

def evaluate_online_checker(args, manager: DataManager, conversational_agent: ConversationalAgent, user_agent: UserAgent):
    max_session_id = manager.set_online_checker(online_ratio=args.online_ratio)
    num_session = min(max_session_id, args.num_session)

    num_turn, success_rate = [], []
    failure_turn = args.num_turn + args.failure_penalty
    for session_id in range(num_session):
        print(f"====== SESSION {session_id} =====")
        data_matrix, label_item_ids, label_attribute_ids = manager.set_session(trainer=conversational_agent.trainer, device=args.device)
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
            print("===== EVALUATE =====")
            print("ID: ", query_id)
            print("RESPONSE: ", response)
            if response is YES_SINGAL:
                num_turn.append(args.num_turn)
                success_rate.append(1)
            else:
                num_turn.append(failure_turn)
                success_rate.append(0)
        
    return (np.mean(num_turn), np.mean(success_rate))


def evaluate_offline_online_loop(args, manager: DataManager, conversational_agent: ConversationalAgent, user_agent: UserAgent, num_loop: int = 1):
    max_session_id = manager.set_online_checker(online_ratio=args.online_ratio)
    num_session = min(max_session_id, args.num_session)
    failure_turn = args.num_turn + args.failure_penalty
    
    for loop_id in range(num_loop):
        dataset = manager.set_offline_trainer(split_ratio=args.split_ratio)
        best_auc = conversational_agent.train(args=args, dataset=dataset)
        print(f"===== AUC at LOOP {loop_id}: {best_auc} ======")

        num_turn, success_rate = [], []
        session_interaction_matrix = []
        for session_id in range(num_session):
            print(f"====== SESSION {session_id} =====")
            data_matrix, label_item_ids, label_attribute_ids = manager.set_session(trainer=conversational_agent.trainer, device=args.device)
            user_id = manager.session2user[session_id]

            conversational_agent.set_session(session_id=session_id, data_matrix=data_matrix)
            user_agent.set_session(session_id=session_id, label_item_ids=label_item_ids, label_attribute_ids=label_attribute_ids)
            
            is_stop =  False
            for turn_id in range(args.num_turn):
                if is_stop:
                    break
                
                query_type, query_id = conversational_agent.check()
                response = user_agent.response(query_type=query_type, query_id=query_id)
                if isinstance(response, tuple): 
                    response_type, response_item_ids = response
                    assert response_type == YES_SINGAL
                    num_turn.append(turn_id)
                    success_rate.append(1)
                    is_stop = True
                elif response == QUIT_SINGAL:
                    num_turn.append(failure_turn)
                    success_rate.append(0)
                    is_stop = True
                else:
                    conversational_agent.set_turn(query_type=query_type, query_id=query_id, response=response)
                
                if query_type == QUERY_ITEM_SIGNAL:
                    for item_id in query_id:
                        if item_id in response_item_ids:
                            session_interaction_matrix.append([user_id, item_id, 1])
                        else:
                            session_interaction_matrix.append([user_id, item_id, 0])
                
            if not is_stop:
                query_id = conversational_agent.evaluate()
                response = user_agent.evaluate(query_id)
                print("===== EVALUATE =====")
                print("ID: ", query_id)
                print("RESPONSE: ", response)
                if response is YES_SINGAL:
                    num_turn.append(args.num_turn)
                    success_rate.append(1)
                else:
                    num_turn.append(failure_turn)
                    success_rate.append(0)
    
        print(f"===== AVG TRUN at LOOP {loop_id}: {np.mean(num_turn)}, AVG SR at LOOP {loop_id}: {np.mean(success_rate)} =====")
        
        session_interaction_matrix = np.array(session_interaction_matrix)
        manager.store(new_interaction_matrix=session_interaction_matrix)
