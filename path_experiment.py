import os
import json
import numpy as np
from utils import *
import argparse
from statsmodels.stats.proportion import proportion_confint
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from unidecode import unidecode
import pickle
import time
import torch
import gc
import socket
import gc
import argparse
import pickle
import copy

import argparse

def get_base_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata_graphs/wikidata_util.json', help='Path to the QA graph, the util.json file')
    parser.add_argument('--context_graph_edge_path', type=str, default='wikidata_graphs/wikidata_text_edge.json', help='Path to the context graph edge file, the text_edge.json file')
    parser.add_argument('--results_dir', type=str, default='final_results/mistral_distractorfull/', help='Directory to save the results')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt', help='Path to the entity aliases file, the entity.txt file')
    parser.add_argument('--id2name_path', type=str, default='wikidata_graphs/wikidata_name_id.json', help='Path to the id2name file, the name_id.json file')
    parser.add_argument('--sentencized_path', type=str, default='wikidata_graphs/wikidata_sentencized.json', help='Path to the sentencized file, the sentencized.json file')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt', help='Path to the relation aliases file, the relation.txt file')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    parser.add_argument('--shuffle_context', action='store_true', default=False, help='Shuffle context in the context of query?')
    parser.add_argument('--k', type=int, default=4, help='Max number of hops in the graph')
    parser.add_argument('--num_queries', type=int, default=250, help='Number of queries to run')
    parser.add_argument('--num_certificates', type=int, default=50, help='Number of certificates to generate')
    return parser

def experiment_pipeline(graph_algos, graph_text_edge, graph_text_sentencized, entity_aliases, source, relation_aliases, id2name, query_model, qa_model, tokenizer, model_context_length, k=5, distractor_query=False, num_queries=5, shuffle_context=True, BATCH_NUM=1, INPUT_DEVICE='cuda:0', rels_path=None):
    assert rels_path is not None, "rels_path cannot be None"
    results = []
    correct = 0
    total = 0
    with torch.no_grad():
        prompts = []
        queries_data = []
        query_data = None
        while query_data is None:
            query_data = get_query_data_path(graph_algos, source, rels_path, id2name, graph_text_edge, graph_text_sentencized, tokenizer, 
                                        distractor_query=distractor_query, k=k, shuffle_context=shuffle_context, max_context_length=model_context_length)
        options_str = '\n'.join([f'{i+1}. {id2name[option]}' for i, option in enumerate(query_data['answer_options'])])
        prompt = LLM_PROMPT_TEMPLATE.format(context=query_data['context'], query=query_data['query'], options=options_str, few_shot_examples=FEW_SHOT_EXAMPLES)
        prompts.append(prompt)
        queries_data.append(query_data)
        if distractor_query:
            assert len(query_data['path_id']) >= 2
        model_answers= query_model(prompts, qa_model, tokenizer, temperature=0.000001, INPUT_DEVICE=INPUT_DEVICE, do_sample=False) #do_sample=False, so no sampling or temp=0
        for i, model_ans in enumerate(model_answers):
            model_ans = model_ans.strip()
            model_answers[i] = model_ans
        assert len(queries_data) == len(model_answers)
        for i in range(len(queries_data)):
            query = queries_data[i]['query']
            correct_answers = queries_data[i]['correct_answers']
            path = queries_data[i]['path_en']
            path_id = queries_data[i]['path_id']
            context = queries_data[i]['context']
            correct_ids = queries_data[i]['correct_ids']
            distractor = queries_data[i]['distractor']
            model_ans = model_answers[i]
            if len(model_ans) == 0:
                continue
            eval_ans = 0
            for num_correct, correct_answer in enumerate(correct_answers):
                eval_ans = dumb_checker(model_ans, queries_data[i]['correct_ans_num'])
                assert id2name[correct_ids[num_correct]] == correct_answer
                if eval_ans == 1:
                    break
            results.append({ 'question':query, 'correct_answers':correct_answers, 'model_answer':model_ans, 
                            'path_en':path, 'path_id':path_id, 'context':context, 'result':(eval_ans, None), 
                            'distractor':distractor, 'correct_ids':correct_ids, 'options':queries_data[i]['answer_options'], 'correct_ans_num':queries_data[i]['correct_ans_num']})
            correct += results[-1]['result'][0]
            total += 1
        del model_answers
    return results

def load_experiment_setup(args, load_model, GPU_MAP):
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, gpu_map=GPU_MAP, quant_type=args.quant_type)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    graph_text_sentencized = json.load(open(args.sentencized_path))
    id2name = json.load(open(args.id2name_path))
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)
    print(f"Best Distractor Task: {args.distractor_query}")
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    REL = ['P17', 'P163']
    best_vertices = []
    vertices_check = qa_graph_algos.get_best_vertices(num=100000)
    for vertex in qa_graph:
        if len(qa_graph_algos.get_queries_for_relpath(REL, vertex))> 0:
            best_vertices.append(vertex)
    print(f"Number of vertices with REL: {len(best_vertices)}")
    #best_vertices =  ['Q38', 'Q1055', 'Q838292', 'Q34433', 'Q254', 'Q31', 'Q270', 'Q200482', 'Q36740', 'Q1911276', 'Q3740786', 'Q1124384', 'Q931739', 'Q2090699', 'Q505788', 'Q1217787', 'Q115448', 'Q2502106', 'Q1793865', 'Q229808', 'Q974437', 'Q219776', 'Q271830', 'Q279164', 'Q76508', 'Q245392', 'Q2546120', 'Q312408', 'Q6110803', 'Q211196', 'Q18407657', 'Q18602670', 'Q21979809', 'Q23010088', 'Q1338555', 'Q5516100', 'Q1765358', 'Q105624', 'Q166262', 'Q33', 'Q36', 'Q16', 'Q96', 'Q36687', 'Q282995', 'Q858401', 'Q850087', 'Q864534', 'Q291244', 'Q159', 'Q668', 'Q211', 'Q183', 'Q1603', 'Q408', 'Q218'][:50]
    random.shuffle(best_vertices)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    return qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices, REL

def get_query_data_path(graph_algos, source, rels_path, id2name, graph_text_edge, graph_text_sentencized, tokenizer, distractor_query=False, k=5, shuffle_context=True, max_context_length=30000):
    while True:
        queries = graph_algos.get_queries_for_relpath(rels_path, source)
        path_ = random.choice(queries)
        start = path[0]
        end = path[-1]
        rel_correct = graph_algos.get_relation_for_vertex(path[-2], path[-1])
        correct_answers = graph_algos.rel2ent_graph[path[-2]][rel_correct]
        query_results = graph_algos.generate_query_for_vertices(start, end, k, path), start, correct_answers, path
        all_distractors = []
        distractor, node_distracted = None, None
        query_inf, _, correct_ids, ids_path = query_results
        query, entity_alias, k_num = query_inf
        path = [id2name[ids_path[i]] for i in range(len(ids_path))]
        if 'country of' in query:
            query = query.replace('country of', 'country location of') # to avoid ambiguity
        true_ids_path = ids_path.copy()
        # if not distractor_query:
        #     random_distractor_parent = random.choice(list(graph_text_edge.keys()))
        #     try:
        #         random_distractor = random.choice(list(graph_text_edge[random_distractor_parent].keys()))
        #     except:
        #         random_distractor = None
        #     if random_distractor not in true_ids_path and random_distractor is not None:
        #         distractor = random_distractor
        #         node_distracted = random_distractor_parent
        options = generate_answer_options(true_ids_path[-1], all_distractors, list(reversed(true_ids_path)), get_random_entities(list(reversed(true_ids_path)), graph_algos.graph), graph_algos.graph)
        relevant_context_list = form_context_list(true_ids_path, graph_text_edge, graph_algos.graph, entity_top_alias=id2name)
        
        relevant_options_context_list = []
        for ent, parent_ent in options:
            relevant_text = graph_text_edge[parent_ent][ent]
            relevant_options_context_list.append(relevant_text)
        
        all_context = get_all_context(true_ids_path + [ent for ent, _ in options], graph_text_sentencized)
        context_list = create_context_list(all_context, relevant_context_list, relevant_options_context_list, tokenizer, max_length=max_context_length)
        if len(context_list) == 0:
            print(f"path ids true: {true_ids_path}, query: {query}, options: {options}")
            raise ValueError("Tokenizer exceeded")
        if distractor is not None:
            ids_path.append(distractor)
            distractor_text = graph_text_edge[node_distracted][distractor]
            context_list.append(distractor_text)
        if shuffle_context:
            random.shuffle(context_list)
        context = '\n'.join([' '.join(context_part) for context_part in context_list])
        if id2name[true_ids_path[0]].lower() != entity_alias.lower():
            context += f" {id2name[true_ids_path[0]]} is also known as {entity_alias}."
        rel_path = [graph_algos.get_relation_for_vertex(true_ids_path[i], true_ids_path[i+1]) for i in range(len(true_ids_path)-1)]
        rel_info = [id2name[rel] for rel in rel_path]
        rel_aliases_used = query.split('->')[1:-1]
        # print(f"rel_info: {rel_info}, rel_aliases_used: {rel_aliases_used}")
        assert len(rel_info) == len(rel_aliases_used)
        rel_context = ' '.join([f"{rel_aliases_used[i]} means the same as {rel_info[i]}" for i in range(len(rel_info))])
        context += f"\n{rel_context}"
        answer_options = [ent for ent, _ in options]
        random.shuffle(answer_options)
        return {'query':query, 'correct_answers':[id2name[correct_id] for correct_id in correct_ids], 'path_id':true_ids_path, 
                'path_en':path, 'context':context, 'correct_ids':correct_ids, 'distractor':distractor, 'answer_options': answer_options, 
                'correct_ans_num': answer_options.index(true_ids_path[-1])+1}
        
def run_experiment(args, load_model, query_model_func, GPU_MAP, model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices, rels_path = load_experiment_setup(args, load_model, GPU_MAP)
    already_done = []
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)
    all_times = []
    num_queries = 0
    experiment_results = []
    start_time = time.time()
    while num_queries < args.num_queries:
        vertex_id = random.choice(best_vertices)
        vertex = id2name[vertex_id]
        start_time = time.time()
        expt_results = experiment_pipeline(graph_algos=qa_graph_algos, graph_text_edge=context_graph_edge, graph_text_sentencized=graph_text_sentencized, 
                                                 entity_aliases=entity_aliases, source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, 
                                                 query_model=query_model_func, qa_model=qa_model, tokenizer=tokenizer, k=args.k, 
                                                 distractor_query=args.distractor_query, num_queries=args.num_queries, 
                                                 shuffle_context=args.shuffle_context, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=model_context_length, rels_path=rels_path)
        experiment_results.extend(expt_results)
        end_time = time.time()
        num_queries += 1
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(experiment_results, f)
    return all_times, num_queries