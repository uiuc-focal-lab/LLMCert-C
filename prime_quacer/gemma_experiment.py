import numpy as np
import argparse
import torch
import google.generativeai as genai
import time
from experiment_utils import *
from experiment_utils import get_base_args, run_experiment
from subgraph_utils import *
from discovery_functions import *

BATCH_NUM = 1
GPU_MAP = {0: "10GiB", 1: "0GiB", 2: "10GiB", 3: "10GiB", "cpu":"0GiB"}
INPUT_DEVICE = 'cuda:0'
CONTINUOUS_SAFE = 0
NUM_GEN = 0
MAX_CONTEXT_LEN = 90000

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='google/gemma-3-4b-it')
    parser.add_argument('--quant_type', type=str, default=None, 
                       choices=['8_bit', '4_bit'], help='No quantization for API models')
    parser.set_defaults(num_queries=250)
    return parser.parse_args()

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = tokenizer.tokenizer
    if not only_tokenizer:
        if quant_type is not None:
           raise NotImplementedError("Quantization not implemented in this example.")
        else:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager",
                device_map="auto", max_memory=gpu_map
            ).eval()
            # model = model.to(INPUT_DEVICE)
        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.pad_token_id
        # assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        # tokenizer.padding_side = 'right'
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # preprocess prompts:
    global MAX_CONTEXT_LEN
    LLAMA3_SYS_PROMPT = "You are a helpful chatbot who answers multiple choice reasoning questions"
    TOT_PROMPT = """
    Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...

Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is...

Identify and behave as three different experts that are appropriate to answering this question.
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus solution or your best guess solution.
The question is:

    """
    chats = []
    if len(prompts) > 1:
        for prompt in prompts:
            message_template = [{"role": "system", "content": [{"type":"text", "text": LLAMA3_SYS_PROMPT}]}, 
                                {"role":"user", "content":[{"type":"text", "text": f"{prompt}"}]}]
            chats.append([copy.deepcopy(message_template)])
    else:
        chats = [{"role": "system", "content": [{"type":"text", "text": LLAMA3_SYS_PROMPT}]}, {"role":"user", "content":[{"type":"text", "text": f"{prompts[0]}"}]}]
        
    input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", add_generation_prompt=True).to(INPUT_DEVICE)
    if input_ids.shape[-1] > MAX_CONTEXT_LEN:
        print("Input too long, input too long, number of tokens: ", input_ids.shape)
        input_ids = input_ids[:, :MAX_CONTEXT_LEN]
    tot_max_length = 24000
    generated_ids= model.generate(input_ids, max_new_tokens=tot_max_length, do_sample=do_sample, temperature=temperature)
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids
    torch.cuda.empty_cache()
    return responses

def main():
    args = get_args()
    
    # Define all discovery functions and their names
    discovery_funcs = [
        lambda graph, **kwargs: off_label_discoverer(graph, **kwargs),
        lambda graph, **kwargs: dual_indication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: synergistic_discoverer(graph, **kwargs),
        # lambda graph, **kwargs: gene_target_discoverer(graph, **kwargs),
        lambda graph, **kwargs: phenotype_drug_contraindication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: drug_contraindication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: exposure_drug_discoverer(graph, **kwargs),
        lambda graph, **kwargs: phenotype_group_disease_discoverer(graph, **kwargs),
        lambda graph, **kwargs: least_side_effects_discoverer(graph, **kwargs),
        lambda graph, **kwargs: contraindication_indication_discoverer(graph, **kwargs)
    ]

    discovery_names = [
        'off_label',
        'dual_indication',
        'synergistic',
        'phenotype_drug_contraindication',
        'drug_contraindication',
        'exposure_drug',
        'phenotype_group_disease',
        'least_side_effects',
        'contraindication_indication'
    ]
    
    # Can specify which certificates to generate:
    # discovery_idx = [0, 2]  # Only generate certificates for specific functions
    # discovery_idx = 0       # Generate certificate for a single function
    discovery_idx = None      # Generate all missing certificates
    
    # Run experiment
    results = run_experiment(
        args,
        load_model=load_model,
        query_model_func=query_model,
        discovery_funcs=discovery_funcs,
        discovery_names=discovery_names,
        GPU_MAP=GPU_MAP,
        BATCH_NUM=BATCH_NUM,
        INPUT_DEVICE=INPUT_DEVICE,
        model_context_length=12800,
        discovery_idx=discovery_idx
    )
    
    # Print results
    print("\nExperiment Results:")
    for func_name, result in results.items():
        if result["completed"]:
            print(f"{func_name}: Generated successfully - Time: {result['time']:.2f} seconds")
        else:
            print(f"{func_name}: Already existed - Skipped")

if __name__ == "__main__":
    main()

