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
GPU_MAP = {0: "20GiB", 1: "15GiB", 2: "0GiB", 3: "0GiB", "cpu":"0GiB"}
INPUT_DEVICE = 'cuda:0'
CONTINUOUS_SAFE = 0
NUM_GEN = 0

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='microsoft/Phi-3-mini-128k-instruct',
                       help='Model name for huggingface transformers')
    parser.add_argument('--quant_type', type=str, default=None, 
                       choices=['8_bit', '4_bit'], help='No quantization for API models')
    parser.set_defaults(num_queries=250)
    return parser.parse_args()

def load_model(model_name="microsoft/Phi-3-mini-128k-instruct", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not only_tokenizer:
        if quant_type is not None:
            if quant_type == '8_bit':
                print("loading 8 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, load_in_8bit=True, trust_remote_code=True, attn_implementation='flash_attention_2')
            elif quant_type == '4_bit':
                print("loading 4 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, bnb_4bit_quant_type="nf4", load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16, trust_remote_code=True, attn_implementation='flash_attention_2')
        else:    
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation='flash_attention_2')
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
        #model.to(INPUT_DEVICE)
        # assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    global NUM_GEN
    NUM_GEN += 1
    # preprocess prompts:
    start_time = time.time()
    PHI_SYS_PROMPT = "You are a helpful AI assistant. who answers multiple choice reasoning questions in a specified format choosing from only the options available"
    chats = []
    if len(prompts) > 1:
        for prompt in prompts:
            message_template = [{"role": "system", "content": PHI_SYS_PROMPT}, {"role":"user", "content":f"{prompt}"}]
            chats.append([copy.deepcopy(message_template)])
    else:
        chats = [{"role": "system", "content": PHI_SYS_PROMPT}, {"role":"user", "content":f"{prompts[0]}"}]
        
    input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", add_generation_prompt=True, padding=True).to(INPUT_DEVICE)
    if input_ids.shape[-1] > 128000:
        print("Input too long, input too long, number of tokens: ", input_ids.shape)
        input_ids = input_ids[:, :128000]
    
    NUM_GEN += 1
    torch.cuda.empty_cache()
    if NUM_GEN % 50 == 0:
        gc.collect()

    start_t = time.time()
    generated_ids= model.generate(input_ids, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    print(f"Time taken for generating: {time.time() - start_t}")
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids
    # print(responses)
    # torch.cuda.empty_cache()
    print(f"Time taken for generating: {time.time() - start_time}")
    return responses

def main():
    args = get_args()
    
    # Define all discovery functions and their names
    discovery_funcs = [
        lambda graph, **kwargs: off_label_discoverer(graph, **kwargs),
        lambda graph, **kwargs: dual_indication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: synergistic_discoverer(graph, **kwargs),
        lambda graph, **kwargs: gene_target_discoverer(graph, **kwargs),
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
        'gene_target',
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