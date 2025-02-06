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
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
import argparse
import google.generativeai as genai

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "10GiB", 1: "10GiB", 2: "10GiB", 3: "10GiB", "cpu":"50GiB"}
INPUT_DEVICE = 'cuda:0'
CONTINUOUS_SAFE = 0
NUM_GEN = 0
MAX_CONTEXT_LEN = 12000
NUM_GEN = 0
def get_base_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_graph_path', type=str, default='quacer_c_prime/graph.json', 
                       help='Path to the QA graph JSON file')
    parser.add_argument('--context_graph_edge_path', type=str, default='quacer_c_prime/graph_text_edge.json',
                       help='Path to the context graph edge file')
    parser.add_argument('--results_dir', type=str, default='baseline/geminiflash/', help='Directory to save the results')
    parser.add_argument('--entity_aliases_path', type=str, default='quacer_c_prime/entity_aliases.txt',
                       help='Path to entity aliases file')
    parser.add_argument('--relation_aliases_path', type=str, default='quacer_c_prime/relation_aliases.txt',
                       help='Path to relation aliases file')
    parser.add_argument('--id2name_path', type=str, default='quacer_c_prime/id2name.json',
                       help='Path to id2name mapping file')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    parser.add_argument('--shuffle_context', action='store_true', default=False, help='Shuffle context in the context of query?')
    parser.add_argument('--k', type=int, default=4, help='Max number of hops in the graph')
    parser.add_argument('--num_queries', type=int, default=50, help='Number of queries to run')
    parser.add_argument('--num_certificates', type=int, default=50, help='Number of certificates to generate')
    parser.add_argument('--qa_llm', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=['8_bit', '4_bit'], help='quantization mode')
    parser.add_argument('--questions_dir', type=str, default='resultsbaseline/llama', help='Directory to load the questions')
    return parser.parse_args()

def experiment_pipeline(questions_data, graph_algos, graph_text_edge, entity_aliases, source, relation_aliases, id2name, query_model, qa_model, tokenizer, model_context_length, k=5, distractor_query=False, num_queries=5, shuffle_context=True, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    results = []
    correct = 0
    total = 0
    num_queries = len(questions_data)
    with torch.no_grad():
        for num_iter in range(num_queries):
            query_data = questions_data[num_iter]
            queries_data = []
            prompts = []
            options_str = '\n'.join([f'{i+1}. {id2name[option]}' for i, option in enumerate(query_data['options'])])
            # print(query_data['question'])
            prompt = LLM_PROMPT_TEMPLATE.format(context=query_data['context'], query=query_data['question'], options=options_str, few_shot_examples=FEW_SHOT_EXAMPLES)
            prompts.append(prompt)
            queries_data.append(query_data)
            model_answers= query_model(prompts, qa_model, tokenizer, temperature=0.000001, INPUT_DEVICE=INPUT_DEVICE, do_sample=False)
            for i, model_ans in enumerate(model_answers):
                model_ans = model_ans.strip()
                model_answers[i] = model_ans
            for i in range(len(queries_data)):
                query = queries_data[i]['question']
                correct_answers = queries_data[i]['correct_answers']
                path = queries_data[i]['path_en']
                path_id = queries_data[i]['path_id']
                context = queries_data[i]['context']
                correct_ids = queries_data[i]['correct_ids']
                distractor = None
                model_ans = model_answers[i]
                if len(model_ans) == 0:
                    continue
                eval_ans = 0
                for num_correct, correct_answer in enumerate(correct_answers):
                    eval_ans = dumb_checker(model_ans, queries_data[i]['correct_ans_num'])
                results.append({ 'question':query, 'correct_answers':correct_answers, 'model_answer':model_ans,
                                'path_en':path, 'path_id':path_id, 'context':context, 'result':(eval_ans, None),
                                'distractor':distractor, 'correct_ids':correct_ids, 'options':queries_data[i]['options'], 'correct_ans_num':queries_data[i]['correct_ans_num']})
                correct += results[-1]['result'][0]
                total += 1
        print(f'Completed {num_queries} queries')
        print(f'Correct = {correct}, Total = {total}')
    return results, correct, total

def load_experiment_setup(args, load_model, GPU_MAP):
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, gpu_map=GPU_MAP, quant_type=args.quant_type)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    graph_text_sentencized = None
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
    for file in os.listdir(args.questions_dir):
        vertex_id = file[:-4]
        vertex = vertex_id
        if vertex_id in already_done:
            print("Already done", vertex_id, vertex)
            continue
        questions_data = pickle.load(open(os.path.join(args.questions_dir, file), 'rb'))[0]
        start_time = time.time()
        print(vertex, vertex_id, len(questions_data))
        num_certificates_generated += 1
        
        results = experiment_pipeline(questions_data=questions_data, graph_algos=qa_graph_algos, graph_text_edge=context_graph_edge, 
                                                 entity_aliases=entity_aliases, source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, 
                                                 query_model=query_model_func, qa_model=qa_model, tokenizer=tokenizer, k=args.k, 
                                                 distractor_query=args.distractor_query, num_queries=args.num_queries, 
                                                 shuffle_context=args.shuffle_context, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=model_context_length)
        end_time = time.time()
        print(f'Time taken for {vertex_id} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(results, f)
    return all_times, num_certificates_generated

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
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
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, attn_implementation='eager')
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
    global MAX_CONTEXT_LEN
    LLAMA3_SYS_PROMPT = "You are a helpful chatbot who answers multiple choice reasoning questions in a specified format choosing from only the options available"
    chats = []
    if len(prompts) > 1:
        for prompt in prompts:
            message_template = [{"role": "system", "content": LLAMA3_SYS_PROMPT}, {"role":"user", "content":f"{prompt}"}]
            chats.append([copy.deepcopy(message_template)])
    else:
        chats = [{"role": "system", "content": LLAMA3_SYS_PROMPT}, {"role":"user", "content":f"{prompts[0]}"}]
        
    input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", add_generation_prompt=True, padding=True).to(INPUT_DEVICE)
    if input_ids.shape[-1] > MAX_CONTEXT_LEN:
        print("Input too long, input too long, number of tokens: ", input_ids.shape)
        input_ids = input_ids[:, :MAX_CONTEXT_LEN]
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