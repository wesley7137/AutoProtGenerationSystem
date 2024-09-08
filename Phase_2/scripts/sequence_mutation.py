import random
import torch

def mutate_sequence(sequence, mutation_rate=0.05):
    mutated_sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice('ACDEFGHIKLMNPQRSTVWY'.replace(mutated_sequence[i], ''))
    return ''.join(mutated_sequence)

def mutate_sequence_bert(sequence, logits, tokenizer, mutation_rate=0.05, threshold=0.5):
    mutated_sequence = list(sequence)
    for i in range(min(len(sequence), logits.size(1))):
        if random.random() < mutation_rate:
            probs = torch.softmax(logits[0, i], dim=-1)
            if probs.max() < threshold:
                mutated_sequence[i] = random.choice('ACDEFGHIKLMNPQRSTVWY'.replace(mutated_sequence[i], ''))
            else:
                top_aa_indices = probs.topk(5).indices.tolist()
                top_aa = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in top_aa_indices]
                top_aa = [aa for aa in top_aa if aa in 'ACDEFGHIKLMNPQRSTVWY' and aa != mutated_sequence[i]]
                if top_aa:
                    mutated_sequence[i] = random.choice(top_aa)
    return ''.join(mutated_sequence)

def mutate_sequence_advanced(sequence, logits, protbert_analyzer, iteration, max_iterations, mutation_rate=0.05):
    mutated_sequence = list(sequence)
    adaptive_rate = mutation_rate * (1 - iteration / max_iterations)
    
    for i in range(min(len(sequence), logits.size(1))):
        if random.random() < adaptive_rate:
            probs = torch.softmax(logits[0, i], dim=-1)
            top_aa_indices = probs.topk(5).indices.tolist()
            top_aa = protbert_analyzer.convert_ids_to_tokens(top_aa_indices)
            
            valid_aa = [aa for aa in top_aa if aa != mutated_sequence[i]]
            
            if valid_aa:
                valid_indices = [idx for idx, aa in zip(top_aa_indices, top_aa) if aa in valid_aa]
                weights = probs[valid_indices].tolist()
                mutated_sequence[i] = random.choices(valid_aa, weights=weights, k=1)[0]
            else:
                mutated_sequence[i] = random.choice('ACDEFGHIKLMNPQRSTVWY'.replace(mutated_sequence[i], ''))
    
    return ''.join(mutated_sequence)