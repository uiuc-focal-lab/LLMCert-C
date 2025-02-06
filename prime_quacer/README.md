# Prime QuaCer: Certifying Knowledge Comprehension in LLMs for Precision Medicine

This folder contains code and resources for certifying knowledge comprehension abilities of Large Language Models (LLMs) within the precision medicine domain using the PrimeKG knowledge graph.

## Overview

Prime QuaCer enables certification of LLMs' ability to answer multi-hop reasoning queries over medical knowledge with formal probabilistic guarantees. The certification framework:

- Uses PrimeKG's graph structure to generate challenging queries spanning drugs, diseases, genes, and phenotypes
- Creates specifications based on common clinical reasoning patterns like drug-disease interactions and phenotype-based diagnosis
- Provides statistical guarantees on an LLM's probability of correctly answering queries sampled from these specifications

**Key Features:**
- Nine reference specifications capturing important clinical reasoning patterns
- Support for both vanilla and distractor settings to test robustness 
- Works with any text-to-text LLM (both open and closed source)
- Provides certification bounds with formal confidence guarantees

## Getting Started

### Prerequisites

- Python 3.7+
- Required Packages:
  ```bash
  pip install -r requirements.txt
  ```
- For open source models: GPU with sufficient VRAM (tested on A100 40GB)
- For closed source models: Appropriate API keys

### Data Setup 

1. Download and preprocess PrimeKG:
   - Download raw PrimeKG data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
   - Follow preprocessing steps in `prime_analysis.ipynb`

2. Required files after preprocessing:
   - `data/kg_util.json`: Final preprocessed knowledge graph
   - `data/kg_text_edge.json`: Supporting text for edges
   - `data/kg_name_id.json`: Entity name mappings
   - `data/entity_aliases.txt`: Entity aliases for disambiguation
   - `data/relation_aliases.txt`: Relation aliases for disambiguation

### Running Experiments

Use one of the provided experiment scripts based on your model:

```bash
python {model}_experiment.py \
    --qa_llm {model_name} \
    --qa_graph_path data/kg_util.json \
    --context_graph_edge_path data/kg_text_edge.json \
    --results_dir results/ \
    --num_queries 250 \
    --num_certificates 10
```

Available experiment scripts:
- `gemini_experiment.py`: For Gemini models
- `gpt_experiment.py`: For GPT models 
- `mistral_experiment.py`: For Mistral models
- `llama_experiment.py`: For Llama models
- `phi_experiment.py`: For Phi models

Key Arguments:
- `--qa_llm`: Name/path of LLM to certify
- `--quant_type`: (Optional) Quantization type for open source models
- `--distractor_query`: Enable distractor-based certification
- `--shuffle_context`: Shuffle context in prompts
- `--num_queries`: Number of queries per certificate
- `--num_certificates`: Total certificates to generate

### Understanding the Code

Key modules:
- `utils.py`: Core functions for graph operations and prompt generation
- `subgraph_utils.py`: Functions for subgraph extraction and manipulation
- `discovery_functions.py`: Definition of clinical reasoning patterns
- `experiment_utils.py`: Main experimental pipeline

See `utilsREADME.md` for details on key utilities and functions.

### Custom Experiments

1. Create experiment file from `model_experiment_template.py`
2. Implement required functions:
   - `get_args()`: Parse model-specific arguments 
   - `load_model()`: Initialize model and tokenizer
   - `query_model()`: Handle prompt batching and generation

### Custom Specifications

To create new clinical reasoning patterns:

1. Define pattern in `discovery_functions.py`:
   - Specify reference DAG structure
   - Define entity type constraints
   - Add template query formats
   - Implement logic for finding isomorphic subgraphs

2. Update discovery function list in experiment script:
   ```python
   discovery_funcs = [
       lambda graph, **kwargs: your_new_pattern(graph, **kwargs),
       ...existing patterns...
   ]
   ```