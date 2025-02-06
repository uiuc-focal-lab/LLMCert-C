from subgraph_utils import CustomQueryResult
import random
import math

def get_non_essential_edges(graph, node, essential_rel_types):
    """Get edges with relation types not used in the essential context"""
    edges = []
    if node in graph:
        for target, rel in graph[node].items():
            if rel not in essential_rel_types:
                edges.append((node, target))
    return edges

def get_essential_edges(graph, node, essential_rel_types):
    """Get edges with relation types used in the essential context"""
    edges = []
    distractor_nodes = []
    if node in graph:
        for target, rel in graph[node].items():
            if rel in essential_rel_types:
                edges.append((node, target))
    random.shuffle(edges)
    for parent, child in edges:
        distractor_nodes.append((child, parent)) # for option generation so answer, parent format
    return edges, distractor_nodes

def build_context_edges(graph, essential_edges, context_nodes, distractor_setting=False, max_context_edges=250):
    """Build context edges based on setting"""
    essential_rel_types = set()
    for n1, n2 in essential_edges:
        if n1 in graph and n2 in graph[n1]:
            essential_rel_types.add(graph[n1][n2])
            essential_rel_types.add(graph[n2][n1]) # undirected graph
    
    context_edges = essential_edges.copy()
    nodes_process = list(context_nodes.copy())
    necessary_nodes = set()
    for n1, n2 in essential_edges:
        necessary_nodes.add(n1)
        necessary_nodes.add(n2)
    necessary_nodes = list(necessary_nodes)
    # Add non-essential context based on setting
    distractor_nodes = []
    if distractor_setting:
        # Include all edges for distractor setting
        random.shuffle(necessary_nodes)
        for node in necessary_nodes:
            if len(context_edges) >= max_context_edges:
                break
            essential_edges, dist_nodes = get_essential_edges(graph, node, essential_rel_types)
            max_edge_collect = min(math.ceil((max_context_edges - len(context_edges))//(len(necessary_nodes)*2)), len(essential_edges))
            if max_edge_collect == 0 and len(essential_edges) > 0:
                max_edge_collect = 1
            essential_edges = essential_edges[:max_edge_collect]
            dist_nodes = dist_nodes[:max_edge_collect]
            context_edges.extend(essential_edges)
            context_edges = list(set(context_edges))
            distractor_nodes.extend(dist_nodes)
        random.shuffle(nodes_process)
        for node in nodes_process:
            if node in necessary_nodes:
                continue
            if len(context_edges) >= max_context_edges:
                    break
            essential_edges, dist_nodes = get_essential_edges(graph, node, essential_rel_types)
            max_edge_collect = min(math.ceil((max_context_edges - len(context_edges))//(len(necessary_nodes)*2)), len(essential_edges))
            if max_edge_collect == 0 and len(essential_edges) > 0:
                max_edge_collect = 1
            essential_edges = essential_edges[:max_edge_collect]
            dist_nodes = dist_nodes[:max_edge_collect]
            context_edges.extend(essential_edges)
            context_edges = list(set(context_edges))
            distractor_nodes.extend(dist_nodes)
        if len(distractor_nodes) < 2:
            print("Distractor nodes less than 2")
            print("Context nodes: ", nodes_process, "\n", necessary_nodes)
            print("Essential edges: ", essential_edges)
            print("Essential rel types: ", essential_rel_types)
            
    random.shuffle(nodes_process)
    for node in nodes_process:
        if len(context_edges) >= max_context_edges:
            break
        non_essential_edges = get_non_essential_edges(graph, node, essential_rel_types)
        non_essential_edges = non_essential_edges[:max_context_edges - len(context_edges)]
        context_edges.extend(non_essential_edges)
        context_edges = list(set(context_edges))
            
    return list(set(context_edges)), list(set(distractor_nodes))  # Remove duplicates

def least_side_effects_discoverer(graph, id2name, **kwargs):
    """Enhanced version with better edge selection"""
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug used to treat {0} has the least number of side effects?',
    ]

    for disease in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease].items() if rel == 'RIDR36']
        if len(drugs) <= 1:
            continue
            
        drug_side_effects = {}
        essential_edges = []
        context_nodes = {disease}
        
        for drug in drugs:
            side_effects = [node for node, rel in graph[drug].items() if rel == 'RIDR16']
            drug_side_effects[drug] = side_effects
            context_nodes.add(drug)
            context_nodes.update(side_effects)
        
        min_count = min(len(se) for se in drug_side_effects.values())
        min_drugs = [d for d, se in drug_side_effects.items() if len(se) == min_count]
        
        if len(min_drugs) >= 1:
            chosen_drug = random.choice(min_drugs)
            other_correct = [d for d in min_drugs if d != chosen_drug]
            essential_edges = [(disease, chosen_drug)]
            essential_edges.extend([(chosen_drug, se) for se in drug_side_effects[chosen_drug]])
            
            question = random.choice(questions).format(id2name[disease])
            other_drugs = [d for d, se in drug_side_effects.items() if len(se) > min_count]
            sorted(other_drugs, key=lambda x: len(drug_side_effects[x]))
            for drug in other_drugs[:5]:
                essential_edges.append((disease, drug))
                essential_edges.extend([(drug, se) for se in drug_side_effects[drug]])
                if len(essential_edges) >= 100:
                    break
            found_queries.append({
                'question': question,
                'chosen_answer': chosen_drug,
                'essential_edges': essential_edges,
                'context_nodes': context_nodes,
                'other_correct_answers': other_correct,
                'distractor_rel': 'RIDR36'
            })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def contraindication_indication_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug is a contraindication for {0} and indication for {1}, when patient has both diseases?',
    ]

    for drug in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
        if not id2name[drug].startswith('(drug)'):
            continue
        
        indications = [node for node, rel in graph[drug].items() if rel == 'RIDR6']
        if not indications:
            continue
        contraindicated_diseases = [node for node, rel in graph[drug].items() if rel == 'RIDR5']
        if not contraindicated_diseases:
            continue
        
        disease0 = random.choice(contraindicated_diseases)
        disease1 = random.choice(indications)
        
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug, rel_id in graph[disease0].items():
            if candidate_drug == drug:
                continue
            if rel_id == 'RIDR35':
                if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] == 'RIDR6':
                    other_correct.append(candidate_drug)
                    
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def off_label_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and is atleast an off-label use drug for {1}?',
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        off_label_diseases = [node for node, rel in graph[drug].items() 
                            if rel in ['RIDR6', 'RIDR7']]
        if not off_label_diseases:
            continue
            
        disease1 = random.choice(off_label_diseases)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] in ['RIDR6', 'RIDR7']:
                if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                    other_correct.append(candidate_drug)
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def dual_indication_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and indicates {1}?',
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        indica_diseases = [node for node, rel in graph[drug].items() if rel == 'RIDR6']
        if not indica_diseases:
            continue
            
        disease1 = random.choice(indica_diseases)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] == 'RIDR6':
                if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                    other_correct.append(candidate_drug)
        
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def synergistic_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and interacts synergistically with the treatment of {1}?',
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        syn_drugs = [node for node, rel in graph[drug].items() if rel == 'RIDR8']
        if not syn_drugs:
            continue
            
        syn_drug = random.choice(syn_drugs)
        indica_syn_drug = [node for node, rel in graph[syn_drug].items() if rel == 'RIDR6']
        if not indica_syn_drug:
            continue
            
        disease1 = random.choice(indica_syn_drug)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                syner_drugs = [node for node, rel in graph[candidate_drug].items() if rel == 'RIDR8']
                for syner_drug in syner_drugs:
                    if disease1 in graph[syner_drug] and graph[syner_drug][disease1] == 'RIDR6':
                        other_correct.append(candidate_drug)
                        break
                    
        essential_edges = [(disease0, drug), (drug, syn_drug), (syn_drug, disease1)]
        context_nodes = {disease0, disease1, drug, syn_drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers':other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def gene_target_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug targets {0} associated with {1}?',
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        genes = [node for node, rel in graph[disease0].items() if rel == 'RIDR38']
        if not genes:
            continue
            
        gene = random.choice(genes)
        drugs = [node for node, rel in graph[gene].items() if rel == 'RIDR33']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        question = random.choice(questions).format(id2name[gene], id2name[disease0])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if gene in graph[candidate_drug] and graph[candidate_drug][gene] == 'RIDR3':
                other_correct.append(candidate_drug)
        
        essential_edges = [(disease0, gene), (gene, drug)]
        context_nodes = {disease0, gene, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })
        
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def phenotype_drug_contraindication_discoverer(graph, id2name, **kwargs):
    """Discovery function for finding diseases sharing phenotypes where one disease's drug contraindicates the other"""
    MAX_BUFFER = 5
    found_queries = []
    
    # Find phenotypes and their associated diseases
    phenotype_nodes = [node for node in graph.keys() 
                      if id2name[node].startswith('(effect/phenotype)')]
    random.shuffle(phenotype_nodes)
    
    questions = [
        'Which disease is contraindicated by drug indication {0} for {1}, which both show {2}?',
    ]
    
    for phenotype in phenotype_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases with this phenotype
        diseases = [node for node, rel in graph[phenotype].items() if rel == 'RIDR13']
        if len(diseases) < 2:
            continue
        
        random.shuffle(diseases)
        # Check each pair of diseases
        for i in range(len(diseases)):
            disease1 = diseases[i]
            for j in range(i+1, len(diseases)):
                disease2 = diseases[j]
                
                # Get drugs indicated for disease1
                drugs1 = [node for node, rel in graph[disease1].items() if rel == 'RIDR36']
                random.shuffle(drugs1)
                
                for drug in drugs1:
                    # Check if drug contraindicates disease2
                    contraindications = [node for node, rel in graph[drug].items() 
                                      if rel == 'RIDR5']
                    if disease2 in contraindications:
                        question = random.choice(questions).format(
                            id2name[drug],
                            id2name[disease1],
                            id2name[phenotype]
                        )
                        
                        essential_edges = [
                            (phenotype, disease1),
                            (phenotype, disease2),
                            (disease1, drug),
                            (drug, disease2)
                        ]
                        context_nodes = {phenotype, disease1, disease2, drug}
                        other_correct = []
                        for candidate_disea in contraindications:
                            if candidate_disea == disease2:
                                continue
                            if phenotype in graph[candidate_disea] and graph[candidate_disea][phenotype] == 'RIDR12':
                                other_correct.append(candidate_disea)
                        
                        found_queries.append({
                            'question': question,
                            'chosen_answer': disease2,
                            'essential_edges': essential_edges,
                            'context_nodes': context_nodes,
                            'other_correct_answers': other_correct
                        })
                        break  # Found a valid query for this disease pair
                
                if found_queries:  # If we found a query, break the inner loop
                    break
            if found_queries:  # If we found a query, break the outer loop
                break
                    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def drug_contraindication_discoverer(graph, id2name, **kwargs):
    """Discovery function for finding diseases that are treated by one drug but contraindicated by another"""
    MAX_BUFFER = 5
    found_queries = []
    
    drug_nodes = [node for node in graph.keys() 
                  if id2name[node].startswith('(drug)')]
    random.shuffle(drug_nodes)
    
    questions = [
        'Which disease is treated with {0} but contraindicated with {1}?',
    ]
    
    for drug1 in drug_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases indicated by drug1
        indications = [node for node, rel in graph[drug1].items() if rel == 'RIDR6']
        if not indications:
            continue
        
        random.shuffle(indications)
        
        # Get other drugs that contraindicate these diseases
        for disease in indications:
            contraindicated_by = []
            for drug2, drug2_data in graph.items():
                if drug2 == drug1:
                    continue
                if disease in [node for node, rel in drug2_data.items() if rel == 'RIDR5']:
                    contraindicated_by.append(drug2)
                    
            if contraindicated_by:
                drug2 = random.choice(contraindicated_by)
                question = random.choice(questions).format(
                    id2name[drug1],
                    id2name[drug2]
                )
                
                essential_edges = [
                    (drug1, disease),
                    (drug2, disease)
                ]
                context_nodes = {drug1, drug2, disease}
                
                other_correct = []
                for candidate_disea in indications:
                    if candidate_disea == disease:
                        continue
                    if drug2 in graph[candidate_disea] and graph[candidate_disea][drug2] == 'RIDR35':
                        other_correct.append(candidate_disea)
                        
                found_queries.append({
                    'question': question,
                    'chosen_answer': disease,
                    'essential_edges': essential_edges,
                    'context_nodes': context_nodes,
                    'other_correct_answers': other_correct
                })
                
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def exposure_drug_discoverer(graph, id2name, **kwargs):
    """Find drugs that treat diseases caused by specific exposures"""
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which drug can be used to treat a disease caused by exposure of {0}?',
    ]

    exposure_nodes = [node for node in graph.keys() 
                     if id2name[node].startswith('(exposure)')]
    random.shuffle(exposure_nodes)
    
    for exposure in exposure_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases caused by this exposure
        exposure_diseases = [node for node, rel in graph[exposure].items() 
                           if rel == 'RIDR24']
        
        for disease in exposure_diseases:
            # Get drugs that treat these diseases
            drugs = [node for node, rel in graph[disease].items() 
                    if rel == 'RIDR36']
            
            if drugs:
                drug = random.choice(drugs)
                question = random.choice(questions).format(id2name[exposure])
                
                essential_edges = [
                    (exposure, disease),
                    (disease, drug)
                ]
                context_nodes = {exposure, disease, drug}
                
                other_correct = []
                for new_drug in drugs:
                    if new_drug == drug:
                        continue
                    for pos_disea, rel_id in graph[new_drug].items():
                        if rel_id == 'RIDR6':
                            if exposure in graph[pos_disea] and graph[pos_disea][exposure] == 'RIDR42':
                                other_correct.append(new_drug)
                                break
                
                found_queries.append({
                    'question': question,
                    'chosen_answer': drug,
                    'essential_edges': essential_edges,
                    'context_nodes': context_nodes,
                    'other_correct_answers': other_correct
                })
                break
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def enzyme_drug_disease_discoverer(graph, id2name, **kwargs):
    """Find diseases treated by drugs that are catalyzed by specific enzymes"""
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which disease is treated by a drug that is catalyzed by the {0}?',
    ]

    enzyme_nodes = [node for node in graph.keys() 
                   if id2name[node].startswith('(gene/enzyme)')]
    random.shuffle(enzyme_nodes)
    
    for enzyme in enzyme_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Find drugs catalyzed by this enzyme
        catalyzed_drugs = []
        for drug, drug_data in graph.items():
            if not id2name[drug].startswith('(drug)'):
                continue
                
            if enzyme in [node for node, rel in drug_data.items() 
                         if rel in ['RIDR2', 'RIDR3', 'RIDR4']]:
                # Check if drug treats any diseases
                diseases = [node for node, rel in drug_data.items() 
                          if rel == 'RIDR6']
                if diseases:
                    catalyzed_drugs.append((drug, diseases))
        
        if catalyzed_drugs:
            drug, diseases = random.choice(catalyzed_drugs)
            disease = random.choice(diseases)
            question = random.choice(questions).format(id2name[enzyme])
            
            essential_edges = [
                (enzyme, drug),
                (drug, disease)
            ]
            context_nodes = {enzyme, drug, disease}
            
            other_correct = []
            for candidate_drug, candidate_diseases in catalyzed_drugs:
                if candidate_drug == drug:
                    continue
                if disease in candidate_diseases:
                    other_correct.append(candidate_drug)
                    
            found_queries.append({
                'question': question,
                'chosen_answer': disease,
                'essential_edges': essential_edges,
                'context_nodes': context_nodes,
                'other_correct_answers': other_correct
            })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def phenotype_group_disease_discoverer(graph, id2name, **kwargs):
    """
    Find diseases that uniquely match a set of phenotypes, but now we also keep track
    of other diseases if the group somehow matches more than one. 
    (Though 'unique' means probably only one disease is truly correct.)
    """
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which disease can be a diagnosis for the following phenotypes - {0}?'
    ]
    
    # Build disease-phenotype mapping
    disease_phenotypes = {}
    for phenotype, phenotype_data in graph.items():
        if not id2name[phenotype].startswith('(effect/phenotype)'):
            continue
            
        diseases = [node for node, rel in phenotype_data.items() if rel == 'RIDR13']
        for disease in diseases:
            if disease not in disease_phenotypes:
                disease_phenotypes[disease] = set()
            disease_phenotypes[disease].add(phenotype)
    
    # Group diseases by their phenotype sets
    phenotype_groups = {}
    for disease, phenotypes in disease_phenotypes.items():
        phenotype_tuple = tuple(sorted(phenotypes))
        phenotype_groups.setdefault(phenotype_tuple, []).append(disease)
    
    # We want groups that identify exactly one disease, but let's see if any group identifies multiple
    # If the group has multiple diseases, they're all correct answers (?). 
    # The question says "Which disease can be a diagnosis for the following phenotypes..."
    # We'll allow multiple diseases, storing them in other_correct_answers.
    random_groups = list(phenotype_groups.items())
    random.shuffle(random_groups)
    
    for phenos, diseases in random_groups:
        if len(found_queries) >= MAX_BUFFER:
            break
        
        if not diseases:
            continue
        # pick one disease
        chosen_disease = random.choice(diseases)
        other_diseases = [d for d in diseases if d != chosen_disease]
        
        # Format phenotype list nicely
        if len(phenos) == 1:
            phenotype_list = id2name[list(phenos)[0]]
        else:
            pheno_names = [id2name[p] for p in phenos]
            if len(pheno_names) > 1:
                phenotype_list = ', '.join(pheno_names[:-1]) + ' and ' + pheno_names[-1]
            else:
                phenotype_list = pheno_names[0]
        
        question = random.choice(questions).format(phenotype_list)
        
        essential_edges = [(p, chosen_disease) for p in phenos]
        context_nodes = set(phenos) | {chosen_disease}
        
        found_queries.append({
            'question': question,
            'chosen_answer': chosen_disease,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_diseases
        })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

import os
import json
import numpy as np
from utils import *
import argparse
from statsmodels.stats.proportion import proportion_confint
import torch
import gc
from unidecode import unidecode
import pickle
import time
import copy
import string
from subgraph_utils import CustomQueryGenerator, CustomQueryResult

def get_base_args():
    parser = argparse.ArgumentParser('Run Relation-based Certificate experiments')
    parser.add_argument('--qa_graph_path', type=str, default='quacer_c_prime/graph.json', 
                       help='Path to the QA graph JSON file')
    parser.add_argument('--context_graph_edge_path', type=str, default='quacer_c_prime/graph_text_edge.json',
                       help='Path to the context graph edge file')
    parser.add_argument('--results_dir', type=str, default='resultsbaseline/llama/',
                       help='Directory to save results')
    parser.add_argument('--entity_aliases_path', type=str, default='quacer_c_prime/entity_aliases.txt',
                       help='Path to entity aliases file')
    parser.add_argument('--relation_aliases_path', type=str, default='quacer_c_prime/relation_aliases.txt',
                       help='Path to relation aliases file')
    parser.add_argument('--id2name_path', type=str, default='quacer_c_prime/id2name.json',
                       help='Path to id2name mapping file')
    parser.add_argument('--shuffle_context', action='store_true', default=False,
                       help='Shuffle context sentences')
    parser.add_argument('--num_queries', type=int, default=250,
                       help='Number of queries per certificate')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    return parser

def load_experiment_setup(args, load_model, GPU_MAP):
    """Load and initialize all required components for the experiment"""
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=True, 
                                   gpu_map=GPU_MAP, quant_type=args.quant_type)
    
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    id2name = json.load(open(args.id2name_path))
    
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)
    
    # Add names to entity aliases
    for key, value in id2name.items():
        if 'p' in key or 'P' in key or 'R' in key or 'r' in key:
            continue
        entity_aliases[key] = [value]
    
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    
    return (qa_graph_algos, context_graph_edge, entity_aliases, 
            relation_aliases, id2name, qa_model, tokenizer)

def experiment_pipeline(graph_algos, graph_text_edge, entity_aliases, 
                       relation_aliases, id2name, query_generator,
                       query_model, qa_model, tokenizer, model_context_length,
                       discovery_func_idx, num_queries=5, shuffle_context=False, 
                       BATCH_NUM=1, INPUT_DEVICE='cuda:0', distractor_query=False):
    """Run the experiment pipeline for a single discovery function"""
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            prompts = []
            queries_data = []
            
            for j in range(BATCH_NUM):
                query_data = None
                while query_data is None:
                    query_result = query_generator.get_single_query(discovery_func_idx, id2name=id2name, distractor_setting=distractor_query)
                    if query_result is not None:
                        query_data = query_generator.generate_query_data(
                            query_result,
                            id2name,
                            graph_text_edge,
                            tokenizer,
                            shuffle_context=shuffle_context,
                            max_context_length=model_context_length,
                            distractor_query=distractor_query
                        )
                
                options_str = '\n'.join([f'{i+1}. {id2name[option]}' 
                                       for i, option in enumerate(query_data['answer_options'])])
                prompt = LLM_PROMPT_TEMPLATE.format(
                    context=query_data['context'],
                    query=query_data['query'],
                    options=options_str,
                    few_shot_examples=FEW_SHOT_EXAMPLES
                )
                
                prompts.append(prompt)
                queries_data.append(query_data)
            model_answers = ['haha' for i in range(len(prompts))]
            
            for i, model_ans in enumerate(model_answers):
                query_data = queries_data[i]
                model_ans = model_ans.strip()
                
                if len(model_ans) == 0:
                    continue
                    
                eval_ans = 0
                
                results.append({
                    'question': query_data['query'],
                    'correct_answers': query_data['correct_answers'],
                    'model_answer': model_ans,
                    'path_en': query_data['path_en'],
                    'path_id': query_data['path_id'],
                    'context': query_data['context'],
                    'result': (eval_ans, None),
                    'correct_ids': query_data['correct_ids'],
                    'options': query_data['answer_options'],
                    'correct_ans_num': query_data['correct_ans_num'],
                    'other_correct_answers': query_data.get('other_correct_answers', [])
                })
                
                correct += eval_ans
                total += 1
                
            # print(f'Completed {num_iter+1} queries, {correct} correct out of {total} total')
            del model_answers
            
    print(f'Completed all {num_queries} queries')
    print(len(results))
    return results, correct, total

def run_experiment(args, load_model, query_model_func, discovery_funcs, discovery_names, GPU_MAP, 
                  model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0', discovery_idx=None):
    """Run the experiment for specified discovery functions
    
    Args:
        discovery_funcs: List of all discovery functions
        discovery_names: List of names corresponding to discovery functions
        discovery_idx: Optional index or list of indices to run specific functions.
                      If None, runs all functions that don't have certificates yet.
    """
    
    # Load experiment components
    experiment_components = load_experiment_setup(args, load_model, GPU_MAP)
    (qa_graph_algos, context_graph_edge, entity_aliases, 
     relation_aliases, id2name, qa_model, tokenizer) = experiment_components

    query_generator = CustomQueryGenerator(qa_graph_algos, discovery_funcs)

    # Determine which functions to run
    if discovery_idx is None:
        indices = range(len(discovery_funcs))
    elif isinstance(discovery_idx, int):
        indices = [discovery_idx]
    else:
        indices = discovery_idx

    results = {}
    for idx in indices:
        func_name = discovery_names[idx]
        
        func_path = os.path.join(args.results_dir, f'{func_name}.pkl')
        if os.path.exists(func_path):
            print(f"Certificate for {func_name} already exists, skipping...")
            results[func_name] = {"time": None, "completed": False}
            continue

        print(f"\nGenerating certificate for {func_name}")
        start_time = time.time()
        
        experiment_results = experiment_pipeline(
            graph_algos=qa_graph_algos,
            graph_text_edge=context_graph_edge,
            entity_aliases=entity_aliases,
            relation_aliases=relation_aliases,
            id2name=id2name,
            query_generator=query_generator,
            discovery_func_idx=idx,  # Pass the index to pipeline
            query_model=query_model_func,
            qa_model=qa_model,
            tokenizer=tokenizer,
            num_queries=args.num_queries,
            shuffle_context=args.shuffle_context,
            BATCH_NUM=BATCH_NUM,
            INPUT_DEVICE=INPUT_DEVICE,
            model_context_length=model_context_length,
            distractor_query=args.distractor_query
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        
        with open(func_path, 'wb') as f:
            pickle.dump(experiment_results, f)
            
        print(f'Completed certificate for {func_name}')
        print(f'Time taken: {time_taken:.2f} seconds')
        
        results[func_name] = {"time": time_taken, "completed": True}
    
    return results

import numpy as np
import argparse
import torch
import google.generativeai as genai
import time
from subgraph_utils import *

BATCH_NUM = 1
GPU_MAP = {0: "20GiB", 1: "15GiB", 2: "0GiB", 3: "0GiB", "cpu":"0GiB"}
INPUT_DEVICE = 'cuda:0'
CONTINUOUS_SAFE = 0
NUM_GEN = 0
MAX_CONTEXT_LEN = 6000

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--quant_type', type=str, default=None, choices=['8_bit', '4_bit'])  # Explicitly set choices
    parser.set_defaults(num_queries=250) # override if needed
    return parser.parse_args()

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