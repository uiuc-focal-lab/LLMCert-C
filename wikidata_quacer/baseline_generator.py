import os
import json
import numpy as np
from baseline_utils import *
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
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
import argparse

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "0GiB", 1: "0GiB", 2: "40GiB", 3: "40GiB", "cpu":"50GiB"}
INPUT_DEVICE = 'cuda:2'

def get_base_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata_graphs/wikidata_util.json', help='Path to the QA graph, the util.json file')
    parser.add_argument('--context_graph_edge_path', type=str, default='wikidata_graphs/wikidata_text_edge.json', help='Path to the context graph edge file, the text_edge.json file')
    parser.add_argument('--results_dir', type=str, default='questions/llama8bsomemore/', help='Directory to save the results')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt', help='Path to the entity aliases file, the entity.txt file')
    parser.add_argument('--id2name_path', type=str, default='wikidata_graphs/wikidata_name_id.json', help='Path to the id2name file, the name_id.json file')
    parser.add_argument('--sentencized_path', type=str, default='wikidata_graphs/wikidata_sentencized.json', help='Path to the sentencized file, the sentencized.json file')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt', help='Path to the relation aliases file, the relation.txt file')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    parser.add_argument('--shuffle_context', action='store_true', default=False, help='Shuffle context in the context of query?')
    parser.add_argument('--k', type=int, default=4, help='Max number of hops in the graph')
    parser.add_argument('--num_queries', type=int, default=50, help='Number of queries to run')
    parser.add_argument('--num_certificates', type=int, default=50, help='Number of certificates to generate')
    parser.add_argument('--qa_llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=['8_bit', '4_bit'], help='quantization mode')
    return parser.parse_args()

def experiment_pipeline(graph_algos, graph_text_edge, graph_text_sentencized, entity_aliases, source, relation_aliases, id2name, query_model, qa_model, tokenizer, model_context_length, k=5, distractor_query=False, num_queries=5, shuffle_context=True, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    results = []
    correct = 0
    total = 0
    paths_done = set()
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            prompts = []
            queries_data = []
            for j in range(BATCH_NUM):
                query_data = None
                while query_data is None:
                    query_data = get_query_data(graph_algos, source, id2name, graph_text_edge, graph_text_sentencized, tokenizer, 
                                                distractor_query=distractor_query, k=k, shuffle_context=shuffle_context, max_context_length=model_context_length)
                    path_ids_select = tuple(query_data['path_id'])
                    if path_ids_select in paths_done:
                        query_data = None
                    else:
                        paths_done.add(path_ids_select)
                options_str = '\n'.join([f'{i+1}. {id2name[option]}' for i, option in enumerate(query_data['answer_options'])])
                prompt = LLM_PROMPT_TEMPLATE.format(context=query_data['context'], query=query_data['query'], options=options_str, few_shot_examples=FEW_SHOT_EXAMPLES)
                prompts.append(prompt)
                queries_data.append(query_data)
                if distractor_query:
                    assert len(query_data['path_id']) >= 2
            for i in range(len(queries_data)):
                query = queries_data[i]['query']
                correct_answers = queries_data[i]['correct_answers']
                path = queries_data[i]['path_en']
                path_id = queries_data[i]['path_id']
                context = queries_data[i]['context']
                correct_ids = queries_data[i]['correct_ids']
                distractor = queries_data[i]['distractor']
                results.append({ 'question':query, 'correct_answers':correct_answers,
                                'path_en':path, 'path_id':path_id, 'context':context, 
                                'distractor':distractor, 'correct_ids':correct_ids, 'options':queries_data[i]['answer_options'], 'correct_ans_num':queries_data[i]['correct_ans_num']})
        print(f'Completed {num_queries} queries')
    return results, correct, total

def load_experiment_setup(args, load_model, GPU_MAP):
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=True, gpu_map=GPU_MAP, quant_type=args.quant_type)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    graph_text_sentencized = json.load(open(args.sentencized_path))
    id2name = json.load(open(args.id2name_path))
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)
    print(f"Best Distractor Task: {args.distractor_query}")
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    # best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    best_vertices =  ['Q38', 'Q1055', 'Q838292', 'Q34433', 'Q254', 'Q31', 'Q270', 'Q200482', 'Q36740', 'Q1911276', 'Q3740786', 'Q1124384', 'Q931739', 'Q2090699', 'Q505788', 'Q1217787', 'Q115448', 'Q2502106', 'Q1793865', 'Q229808', 'Q974437', 'Q219776', 'Q271830', 'Q279164', 'Q76508', 'Q245392', 'Q2546120', 'Q312408', 'Q6110803', 'Q211196', 'Q18407657', 'Q18602670', 'Q21979809', 'Q23010088', 'Q1338555', 'Q5516100', 'Q1765358', 'Q105624', 'Q166262', 'Q33', 'Q36', 'Q16', 'Q96', 'Q36687', 'Q282995', 'Q858401', 'Q850087', 'Q864534', 'Q291244', 'Q159', 'Q668', 'Q211', 'Q183', 'Q1603', 'Q408', 'Q218'][:50]
    # random.shuffle(best_vertices)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    return qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices

def run_experiment(args, load_model, query_model_func, GPU_MAP, model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices = load_experiment_setup(args, load_model, GPU_MAP)
    already_done = []
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)
    all_times = []
    num_certificates_generated = 0
    for i, vertex_id in enumerate(best_vertices):
        if num_certificates_generated >= args.num_certificates:
            break
        vertex = id2name[vertex_id]
        if vertex_id in already_done:
            print("Already done", vertex_id, vertex)
            continue
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex_id, 4)
        subgraph_algos = GraphAlgos(subgraph, entity_aliases, relation_aliases)
        if len(subgraph) < 900:
            print(len(subgraph), "Skipping", vertex_id, vertex) # Skip small subgraphs
            continue
        print(vertex, vertex_id, len(subgraph))
        num_certificates_generated += 1
        
        questions = experiment_pipeline(graph_algos=subgraph_algos, graph_text_edge=context_graph_edge, graph_text_sentencized=graph_text_sentencized, 
                                                 entity_aliases=entity_aliases, source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, 
                                                 query_model=query_model_func, qa_model=qa_model, tokenizer=tokenizer, k=args.k, 
                                                 distractor_query=args.distractor_query, num_queries=args.num_queries, 
                                                 shuffle_context=args.shuffle_context, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=model_context_length)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        print(os.path.join(args.results_dir, str(vertex_id)+'.pkl'))
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(questions, f)
    return all_times, num_certificates_generated

def load_model(model_name="lmsys/vicuna-13b-v1.5", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not only_tokenizer:
        if quant_type is not None:
            if quant_type == '8_bit':
                print("loading 8 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, load_in_8bit=True)
            elif quant_type == '4_bit':
                print("loading 4 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, bnb_4bit_quant_type="nf4", load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16)
        else:
            print('no quantization, loading in fp16')
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        tokenizer.padding_side = 'right'
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # preprocess prompts:
    LLAMA3_SYS_PROMPT = "You are a helpful chatbot who answers multiple choice reasoning questions in a specified format choosing from only the options available"
    chats = []
    if len(prompts) > 1:
        for prompt in prompts:
            message_template = [{"role": "system", "content": LLAMA3_SYS_PROMPT}, {"role":"user", "content":f"{prompt}"}]
            chats.append([copy.deepcopy(message_template)])
    else:
        chats = [{"role": "system", "content": LLAMA3_SYS_PROMPT}, {"role":"user", "content":f"{prompts[0]}"}]
        
    input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", add_generation_prompt=True, padding=True).to(INPUT_DEVICE)
    if input_ids.shape[-1] > 8000:
        print("Input too long, input too long, number of tokens: ", input_ids.shape)
        input_ids = input_ids[:, :8000]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    generated_ids= model.generate(input_ids, max_new_tokens=max_length, do_sample=do_sample, eos_token_id=terminators, temperature=temperature)
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids

    return responses

def main():
    args = get_base_args()
    print(args)
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=7200)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()