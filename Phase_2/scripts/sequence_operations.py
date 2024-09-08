import random
from Bio import Align
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def crossover(seq1, seq2):
    crossover_point = random.randint(0, len(seq1) - 1)
    return seq1[:crossover_point] + seq2[crossover_point:]

def calculate_conservation_scores(sequences):
    if isinstance(sequences[0], str):
        # Convert strings to SeqRecord objects
        sequences = [SeqRecord(Seq(seq), id=f"seq_{i}") for i, seq in enumerate(sequences)]
    
    alignment = Align.MultipleSeqAlignment(sequences)
    
    # Calculate conservation scores (this is a simple example, you might want to use a more sophisticated method)
    conservation_scores = []
    for i in range(len(alignment[0])):
        column = alignment[:, i]
        score = sum(1 for c in column if c == column[0]) / len(column)
        conservation_scores.append(score)
    
    return conservation_scores
def adaptive_mutation_rate(iteration, max_iterations, current_score, last_improvement):
    initial_rate = 0.05
    if iteration - last_improvement > 10:
        return min(0.5, initial_rate * (1 + (iteration - last_improvement) / 10))
    return initial_rate