import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.stats import boltzmann

import torch
from transformers import BertForMaskedLM, BertTokenizer
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from predict5 import predict_protein_function, predict_properties, predict_structure
import random
import numpy as np
from scipy.stats import boltzmann
import numpy as np
from scipy.stats import boltzmann
import torch
from transformers import BertForMaskedLM, BertTokenizer
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from predict5 import predict_protein_function, predict_properties, predict_structure
import random
from deap import base, creator, tools, algorithms


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load ProtBERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

# Define a dictionary of amino acids with known functions or features
aa_functions = {
    'C': 'disulfide bond formation',
    'D': 'negative charge',
    'E': 'negative charge',
    'K': 'positive charge',
    'R': 'positive charge',
    'H': 'metal binding',
    'S': 'phosphorylation site',
    'T': 'phosphorylation site',
    'Y': 'phosphorylation site',
    'W': 'hydrophobic core',
    'F': 'hydrophobic core',
    'L': 'hydrophobic core',
    'I': 'hydrophobic core',
    'V': 'hydrophobic core',
    'G': 'flexibility',
    'P': 'turn formation'
}

# Monte Carlo approach with simulated annealing
async def monte_carlo_optimize(sequence, iterations=100, initial_temperature=1.0, cooling_rate=0.95):
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)
    best_sequence = current_sequence
    best_score = current_score
    temperature = initial_temperature

    for i in range(iterations):
        # Make a random mutation
        mutated_sequence = list(current_sequence)
        mutation_pos = random.randint(0, len(mutated_sequence) - 1)
        mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
        mutated_sequence = ''.join(mutated_sequence)

        # Evaluate the mutated sequence
        mutated_score = await predict_protein_function(mutated_sequence)

        # Decide whether to accept the new sequence
        delta_score = mutated_score - current_score
        if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
            current_sequence = mutated_sequence
            current_score = mutated_score

            if current_score > best_score:
                best_sequence = current_sequence
                best_score = current_score

        # Cool down the temperature
        temperature *= cooling_rate

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return best_sequence, best_score


# Gradient-based optimization
async def gradient_optimize(sequence, iterations=50, learning_rate=0.01):
    # This is a simplified version. In practice, you'd need a differentiable model for protein function prediction.
    current_sequence = sequence
    for i in range(iterations):
        # Compute "gradient" (this is a placeholder for actual gradient computation)
        gradient = np.random.randn(len(current_sequence))

        # Update sequence based on gradient
        new_sequence = ''
        for j, aa in enumerate(current_sequence):
            aa_index = 'ACDEFGHIKLMNPQRSTVWY'.index(aa)
            new_index = int((aa_index + learning_rate * gradient[j]) % 20)
            new_sequence += 'ACDEFGHIKLMNPQRSTVWY'[new_index]

        current_sequence = new_sequence

        # Evaluate new sequence
        score = await predict_protein_function(current_sequence)
        logger.info(f"Iteration {i+1}: Score {score}")

    return current_sequence, score

# Domain-specific knowledge optimization
async def domain_knowledge_optimize(sequence, iterations=50):
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)

    for i in range(iterations):
        # Apply domain-specific rules (this is a simplified example)
        new_sequence = list(current_sequence)

        # Rule 1: Ensure hydrophobic core
        if 'W' not in new_sequence and 'F' not in new_sequence:
            new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['W', 'F'])

        # Rule 2: Ensure some charged residues
        if 'K' not in new_sequence and 'R' not in new_sequence and 'D' not in new_sequence and 'E' not in new_sequence:
            new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['K', 'R', 'D', 'E'])

        new_sequence = ''.join(new_sequence)
        new_score = await predict_protein_function(new_sequence)

        if new_score > current_score:
            current_sequence = new_sequence
            current_score = new_score

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return current_sequence, current_score

# Ensemble approach
async def ensemble_optimize(sequence, iterations=50):
    methods = [monte_carlo_optimize, gradient_optimize, domain_knowledge_optimize]
    sequences = []
    scores = []

    for method in methods:
        optimized_seq, score = await method(sequence, iterations=iterations//len(methods))
        sequences.append(optimized_seq)
        scores.append(score)

    # Choose the best sequence
    best_index = np.argmax(scores)
    return sequences[best_index], scores[best_index]

# Simplified Reinforcement Learning
class RLAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.q_table = np.zeros((20, action_space_size))  # 20 amino acids

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q

async def rl_optimize(sequence, iterations=1000, epsilon=0.1, alpha=0.1, gamma=0.9):
    agent = RLAgent(20)  # 20 possible actions (amino acids)
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)
    best_sequence = current_sequence
    best_score = current_score

    for i in range(iterations):
        position = random.randint(0, len(current_sequence) - 1)
        current_aa = 'ACDEFGHIKLMNPQRSTVWY'.index(current_sequence[position])
        action = agent.choose_action(current_aa, epsilon)

        new_sequence = list(current_sequence)
        new_sequence[position] = 'ACDEFGHIKLMNPQRSTVWY'[action]
        new_sequence = ''.join(new_sequence)

        new_score = await predict_protein_function(new_sequence)
        reward = new_score - current_score

        agent.update_q_table(current_aa, action, reward, action, alpha, gamma)

        if new_score > current_score:
            current_sequence = new_sequence
            current_score = new_score

            if current_score > best_score:
                best_sequence = current_sequence
                best_score = current_score

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return best_sequence, best_score





# Constraint-based optimization
async def constraint_optimize(sequence, iterations=50, min_hydrophobicity=0.3, max_charge=0.2):
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)
    best_sequence = current_sequence
    best_score = current_score

    for i in range(iterations):
        # Make a random mutation
        mutated_sequence = list(current_sequence)
        mutation_pos = random.randint(0, len(mutated_sequence) - 1)
        mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
        mutated_sequence = ''.join(mutated_sequence)

        # Check constraints
        analysis = ProteinAnalysis(mutated_sequence)
        hydrophobicity = analysis.gravy()
        charge = abs(analysis.charge_at_pH(7.0)) / len(mutated_sequence)

        if hydrophobicity >= min_hydrophobicity and charge <= max_charge:
            # Evaluate the mutated sequence
            mutated_score = await predict_protein_function(mutated_sequence)

            if mutated_score > current_score:
                current_sequence = mutated_sequence
                current_score = mutated_score

                if current_score > best_score:
                    best_sequence = current_sequence
                    best_score = current_score

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return best_sequence, best_score


# Multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    sequence = ''.join(individual)
    score = asyncio.run(predict_protein_function(sequence))
    stability = -1 * ProteinAnalysis(sequence).instability_index()  # Higher stability is better
    return score, stability

def mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice('ACDEFGHIKLMNPQRSTVWY')
    return individual,

async def multi_objective_optimize(sequence, population_size=50, generations=50):
    toolbox = base.Toolbox()
    toolbox.register("attr_char", random.choice, 'ACDEFGHIKLMNPQRSTVWY')
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_char, n=len(sequence))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    best_sequence = ''.join(best_individual)
    best_score, best_stability = evaluate(best_individual)

    return best_sequence, best_score, best_stability

# Diversity preservation
async def diversity_optimize(sequence, population_size=50, generations=50):
    def create_individual():
        return ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(len(sequence)))

    population = [create_individual() for _ in range(population_size)]

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = await asyncio.gather(*[predict_protein_function(ind) for ind in population])

        # Sort population by fitness
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

        # Select top individuals
        elite_size = population_size // 4
        new_population = [ind for ind, _ in sorted_population[:elite_size]]

        # Create new individuals
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = ''.join(p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2))

            # Mutation
            if random.random() < 0.1:
                pos = random.randint(0, len(child) - 1)
                child = child[:pos] + random.choice('ACDEFGHIKLMNPQRSTVWY') + child[pos+1:]

            # Add to population if diverse enough
            if all(tools.hamming(child, ind) > len(sequence) // 10 for ind in new_population):
                new_population.append(child)

        population = new_population

        logger.info(f"Generation {gen+1}: Best score {sorted_population[0][1]}")

    best_sequence = sorted_population[0][0]
    best_score = sorted_population[0][1]
    return best_sequence, best_score


async def run_optimization_pipeline(sequences, iterations=50, score_threshold=0.4):
    optimized_results = []
    print(f"Starting optimization pipeline with sequences: {sequences}")
    for sequence in sequences:
        try:
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            sequence = ''.join(char for char in sequence if char in valid_aa)
            if not sequence:
                logger.warning("Skipping empty or invalid sequence")
                continue

            # Run all optimization methods
            constraint_sequence, constraint_score = await constraint_optimize(sequence, iterations)
            multi_obj_sequence, multi_obj_score, multi_obj_stability = await multi_objective_optimize(sequence, iterations=iterations)
            diversity_sequence, diversity_score = await diversity_optimize(sequence, generations=iterations)
            constrained_mc_sequence, constrained_mc_score = await constrained_monte_carlo_optimize(sequence, iterations)
            adaptive_mc_sequence, adaptive_mc_score = await adaptive_monte_carlo_optimize(sequence, iterations)

            # Choose the best result
            optimized_sequences = [constraint_sequence, multi_obj_sequence, diversity_sequence, constrained_mc_sequence, adaptive_mc_sequence]
            optimized_scores = [constraint_score, multi_obj_score, diversity_score, constrained_mc_score, adaptive_mc_score]
            best_index = np.argmax(optimized_scores)
            optimized_sequence = optimized_sequences[best_index]
            optimized_score = optimized_scores[best_index]

            if optimized_score >= score_threshold:
                # Predict properties for the final optimized sequence
                properties = await predict_properties(optimized_sequence)
                optimized_results.append({
                    "original_sequence": sequence,
                    "optimized_sequence": optimized_sequence,
                    "original_score": await predict_protein_function(sequence),
                    "optimized_score": optimized_score,
                    "properties": properties,
                    "best_method": ["Constraint", "Multi-Objective", "Diversity", "Constrained Monte Carlo", "Adaptive Monte Carlo"][best_index]
                })
                logger.info(f"Sequence optimization successful. Final score: {optimized_score}")
                logger.info(f"Best method: {optimized_results[-1]['best_method']}")
            else:
                logger.info(f"Sequence optimization did not meet threshold. Score: {optimized_score}")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            continue

    return optimized_results


if __name__ == "__main__":
    async def main():
        test_sequences = [
            'ITASAWWRSANRSQQLKWTLLGFTCNMVFFPTAHKVQAHATKWLMAREFYGDFNDLTQRAIGPSGGLADHYPTWGYRLMDATGAPGTTMAFLVASLAVFGALVYVIFVVCFAPMAVKDYVAERKVGPIELMMFDVVTLHLLVPFPLLNAASIIAGVQAGIESWGIVSLGVKVGRFGARIPIGVVAAVRLTWMIPRRPAWSADRMRPEPPGPKVYAYRLFSERPIAFDAMFGAVALLGWLTVWRSRKGRVWPSWMGS'
        ]

        results = await run_optimization_pipeline(test_sequences, iterations=50, score_threshold=0.4)

        for result in results:
            print("\nOptimization Result:")
            print(f"Original Sequence: {result['original_sequence'][:50]}...")
            print(f"Optimized Sequence: {result['optimized_sequence'][:50]}...")
            print(f"Original Score: {result['original_score']}")
            print(f"Optimized Score: {result['optimized_score']}")
            print("Properties:")
            for prop, value in result['properties'].items():
                print(f"  {prop}: {value}")

    asyncio.run(main())