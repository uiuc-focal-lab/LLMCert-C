#!/usr/bin/env python3
"""
Template for creating a new model experiment file for Prime LLMCert-C.

This template demonstrates how to run a certification experiment.
Custom experiment files should define:
 - get_args(): for command line argument parsing.
 - load_model(): to load the LLM and its tokenizer.
 - query_model(): to query the model using prompts.
 - main(): to run the experiment.

Replace the placeholder implementations with your model-specific code.
"""

import argparse

import experiment_utils
import utils  # Import utilities as needed
import numpy as np
import argparse
import torch
import google.generativeai as genai
import time
from experiment_utils import *
from experiment_utils import get_base_args, run_experiment
from subgraph_utils import *
from discovery_functions import *

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prime LLMCert-C Model Experiment Template")
    # Base experiment arguments
    parser.add_argument("--qa_llm", type=str, required=True, help="Name or path of the LLM to be certified.")
    parser.add_argument("--quant_type", type=str, default=None, help="Quantization type (e.g., '8_bit', '4_bit').")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries per certificate.")
    parser.add_argument("--num_certificates", type=int, default=10, help="Total number of certificates to generate.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store results.")
    # Add additional arguments as needed
    return parser.parse_args()


def load_model(model_name, only_tokenizer=False, gpu_map=None, quant_type=None):
    """
    Load the specified model and its tokenizer.

    Parameters:
    - model_name: str, name or path of the model.
    - only_tokenizer: bool, if True, load only the tokenizer.
    - gpu_map: dict, mapping of GPU identifiers.
    - quant_type: str, type of quantization to be used.

    Returns:
    A tuple (model, tokenizer).
    """
    # Placeholder implementation. Replace with actual model loading code.
    print(f"Loading model '{model_name}' with quant_type={quant_type}")
    model = None  # Replace with your model loading logic.
    tokenizer = None  # Replace with your tokenizer loading logic.
    return model, tokenizer


def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    """
    Query the LLM with a list of prompts.

    Parameters:
    - prompts: list of str, input prompts.
    - model: loaded model.
    - tokenizer: loaded tokenizer.
    - Other parameters: control generation settings.

    Returns:
    A list of responses from the model.
    """
    # Placeholder implementation. Replace with your model's inference code.
    responses = []
    for prompt in prompts:
        print(f"Querying model with prompt: {prompt[:50]}...")  # Show a snippet of the prompt
        response = "dummy response"  # Replace with the actual model output.
        responses.append(response)
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
