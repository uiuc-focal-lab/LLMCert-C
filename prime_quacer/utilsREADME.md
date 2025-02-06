# Prime QuaCer Utilities Documentation

This document describes the core utilities for certifying knowledge comprehension capabilities of LLMs using the PrimeKG knowledge graph.

## Core Utilities (utils.py)

### GraphAlgos Class
Provides operations for manipulating and querying medical knowledge graphs:

```python
graph_algos = GraphAlgos(graph, entity_aliases, relation_aliases)
```

Key methods:
- `create_subgraph_within_radius(start_vertex, k)`: Extract subgraph for certificate 
- `get_queries_for_relpath(rel_path, start_vertex)`: Find paths matching relation sequence
- `generate_template_path(path_rels, source_vertices)`: Generate paths matching template
- `get_best_distractor(start_vertex, path)`: Find challenging distractor nodes

### Query Generation
Functions for creating natural language queries from graph paths:

- `generate_answer_options()`: Create answer choices including distractions
- `form_context_list()`: Assemble relevant context for query
- `create_context_list()`: Trim context to fit model limits

## Subgraph Utilities (subgraph_utils.py)

### CustomQueryGenerator
Handles query generation for certification:

```python 
generator = CustomQueryGenerator(graph_algos, discovery_funcs)
```

- `get_single_query()`: Generate query from discovery function
- `generate_query_data()`: Create complete prompt with context/options

### Subgraph Operations
- `get_entity_type()`: Extract entity categories (drug/disease/gene)
- `generate_options_with_exclusions()`: Create answer options preserving types
- `get_random_entities_filter()`: Sample entities respecting constraints

## Discovery Functions (discovery_functions.py)

Each function defines a clinical reasoning pattern:

```python
def pattern_discoverer(graph, id2name, **kwargs):
    """Find subgraphs matching clinical pattern"""
    # Define reference DAG structure
    # Add entity type constraints  
    # Return CustomQueryResult
```

Built-in patterns:
- Drug-disease interaction
- Off-label drug use
- Phenotype-based diagnosis
- Gene-drug targeting
- Drug synergy
- Exposure causation
- Drug contraindication

See function docstrings for detailed specifications.

## Experiment Utilities (experiment_utils.py)

Key functions for running certification:

### Setup
```python
components = load_experiment_setup(args, load_model, GPU_MAP)
```
Loads graph data, model, and initializes certification

### Main Pipeline
```python
results = run_experiment(args, load_model, query_model_func, 
                        discovery_funcs, GPU_MAP)
```
Generates certificates for each specified reasoning pattern

### Evaluation
- `experiment_pipeline()`: Core certification loop for single pattern
- Uses Clopper-Pearson intervals for high-confidence bounds

## Using the Framework

1. Define reasoning pattern in discovery_functions.py
2. Create experiment script from template
3. Run certification with desired settings
4. Analyze results using provided notebooks

See main README for full usage examples.