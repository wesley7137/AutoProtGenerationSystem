import torch
import logging
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, EsmForProteinFolding
from Bio.PDB import PDBParser
from Bio.Seq import Seq

# Import custom modules
from scripts.protbert_analyzer import ProtBERTAnalyzer
from scripts.protein_analysis import predict_protein_function
from scripts.sequence_operations import crossover, adaptive_mutation_rate
from scripts.sequence_mutation import mutate_sequence, mutate_sequence_bert, mutate_sequence_advanced
from scripts.mutation_predictor import SimpleMutationPredictor
from scripts.protbert_analyzer import ProtBERTAnalyzer
from scripts.protein_analysis import predict_protein_function
from scripts.sequence_operations import crossover, calculate_conservation_scores, adaptive_mutation_rate
from scripts.validate_initial_structure import validate_initial_structure
from scripts.optimization_pipeline import protein_engineering, stability_optimization, functional_testing
import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
log_file_path = 'logs/optimization_pipeline.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        RotatingFileHandler(log_file_path, maxBytes=10000000, backupCount=5),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
progen_model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code=True).to(device_1)
progen_tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-large", trust_remote_code=True)

def generate_protein_sequence(input_text, max_length=512):
    progen_model.eval()
    input_text = f"Input Text: {input_text} Sequence:"
    input_ids = progen_tokenizer(input_text, return_tensors='pt').input_ids.to(device_1)
   
    with torch.no_grad():
        output = progen_model.generate(input_ids, max_length=max_length, num_return_sequences=5, do_sample=True, top_p=0.95)
   
    generated_sequence = progen_tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated sequence: {generated_sequence}")
    return generated_sequence.split("Sequence:")[-1].strip()

def structure_to_sequence(structure_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", structure_file)
    logger.info(f"Structure file: {structure_file}")
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Check if it's a standard amino acid
                    sequence += Seq.protein_letters_3to1[residue.get_resname()]
    logger.info(f"Sequence: {sequence}")
    return sequence

def generate_structure_esm3(sequence):
    try:
        device_2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = EsmForProteinFolding.from_pretrained("EvolutionaryScale/esm3-sm-open-v1").to(device_2   )
        tokenizer = AutoTokenizer.from_pretrained("EvolutionaryScale/esm3-sm-open-v1")
        logger.info(f"Input sequence: {sequence}")
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device_2)
        
        with torch.no_grad():
            outputs = model(**inputs)
        logger.info(f"Structure generated. Output shape: {outputs.positions.shape}")
        return outputs, sequence
    except Exception as e:
        logger.error(f"Error in generate_structure_esm3: {str(e)}")
        warnings.warn("ESM3 model loading failed. Proceeding with sequence-based optimization.")
        return None, sequence
    
def sequence_based_optimization(sequence, iteration, max_iterations):
    logger.info(f"Starting sequence-based optimization. Iteration: {iteration}/{max_iterations}")
    logger.info(f"Input sequence: {sequence}")
    
    # Remove invalid characters from the sequence
    sequence = ''.join(char for char in sequence if char in 'ACDEFGHIKLMNPQRSTVWY')
    
    mutation_predictor = SimpleMutationPredictor()
    protbert_analyzer = ProtBERTAnalyzer()
    
    protbert_analyzer.load_model()
    logger.info("ProtBERT model loaded")
    
    logits = protbert_analyzer.analyze_sequence(sequence)
    logger.info(f"Sequence analysis completed. Logits shape: {logits.shape}")
    
    try:
        current_score = predict_protein_function(sequence)
    except Exception as e:
        logger.error(f"Error in predict_protein_function: {str(e)}")
        current_score = 0.5  # Default score if prediction fails
    
    logger.info(f"Current protein function score: {current_score}")
    
    # Assume last_improvement is 0 if it's the first iteration
    last_improvement = 0 if iteration == 0 else iteration - 1
    mutation_rate = adaptive_mutation_rate(iteration, max_iterations, current_score, last_improvement)
    logger.info(f"Adaptive mutation rate: {mutation_rate}")
    
    # Generate multiple candidate sequences
    candidate_sequences = [
        mutate_sequence(sequence, mutation_rate),
        mutate_sequence_bert(sequence, logits, protbert_analyzer.tokenizer, mutation_rate),
        mutate_sequence_advanced(sequence, logits, protbert_analyzer, iteration, max_iterations, mutation_rate)
    ]
    
    # Add crossover sequences
    for i in range(len(candidate_sequences)):
        for j in range(i+1, len(candidate_sequences)):
            candidate_sequences.append(crossover(candidate_sequences[i], candidate_sequences[j]))
    
    logger.info(f"Generated {len(candidate_sequences)} candidate sequences")
    
    # Evaluate all candidates
    scores = []
    for seq in candidate_sequences:
        try:
            score = predict_protein_function(seq)
        except Exception as e:
            logger.error(f"Error in predict_protein_function: {str(e)}")
            score = 0  # Assign a low score if prediction fails
        scores.append(score)
    
    # Select the best sequence
    best_sequence = candidate_sequences[scores.index(max(scores))]
    best_score = max(scores)
    
    logger.info(f"Best sequence score: {best_score}")
    logger.info(f"Best sequence: {best_sequence}")
    
    protbert_analyzer.unload_model()
    logger.info("ProtBERT model unloaded")
    
    return best_sequence, best_score


def optimization_pipeline(input_text, num_cycles=2):
    logger.info(f"Starting optimization pipeline with input: {input_text}")
    sequence = generate_protein_sequence(input_text, max_length=200)
    # Remove invalid characters from the initial sequence
    sequence = ''.join(char for char in sequence if char in 'ACDEFGHIKLMNPQRSTVWY')
    logger.info(f"**SYSTEM** Initial generated sequence: {sequence}")
    final_structure = None
    best_score = float('-inf')
    best_sequence = sequence

    for cycle in range(num_cycles):
        logger.info(f"**SYSTEM** Starting Optimization Cycle {cycle + 1}/{num_cycles}")
        
        # Sequence-based optimization
        optimized_sequence, current_score = sequence_based_optimization(sequence, cycle, num_cycles)
        logger.info(f"**SYSTEM** Optimized sequence: {optimized_sequence}")
        logger.info(f"**SYSTEM** Current function score: {current_score}")
        
        if current_score > best_score:
            best_score = current_score
            best_sequence = optimized_sequence
            logger.info(f"**SYSTEM** New best sequence found. Score: {best_score}")
        
        # Update the sequence for the next cycle
        sequence = optimized_sequence

    # After all optimization cycles, generate the structure for the best sequence
    try:
        final_structure, _ = generate_structure_esm3(best_sequence)
        logger.info(f"**SYSTEM** Final structure generated. Shape: {final_structure.positions.shape}")
        
        # Prepare the input for validate_initial_structure
        validation_input = {
            "generated_protein": final_structure,
            "prompt_string": input_text,
            "results": {}
        }
        
        validation_result = validate_initial_structure(validation_input)
        logger.info(f"**SYSTEM** Validation result: {validation_result}")
        
        if validation_result["status"] == "passed":
            func_results = functional_testing(final_structure)
            logger.info(f"**SYSTEM** Functional testing results: {func_results}")
            
            stab_results = stability_optimization(final_structure)
            logger.info(f"**SYSTEM** Stability optimization results: {stab_results}")
            
            final_structure = protein_engineering(final_structure, func_results, stab_results)
            logger.info(f"**SYSTEM** Final structure optimized. Shape: {final_structure.positions.shape}")
        else:
            logger.warning(f"**SYSTEM** Final structure failed validation. Score: {validation_result['validity_score']}")
    
    except Exception as e:
        logger.error(f"Error in final structure generation: {str(e)}")
        final_structure = None


if __name__ == "__main__":
    input_description = "A peptide that increases growth activity in telomeres"
    logger.info(f"Starting optimization with input description: {input_description}")
    
    final_sequence, final_structure, final_score = optimization_pipeline(input_description)
    logger.info(f"Final optimized protein sequence: {final_sequence}")
    logger.info(f"Final predicted function score: {final_score}")
    logger.info(f"Final structure shape: {final_structure.positions.shape if final_structure else 'No structure generated'}")
    
    with open("optimization_results.txt", "w") as f:
        f.write(f"Input description: {input_description}\n")
        f.write(f"Final sequence: {final_sequence}\n")
        f.write(f"Function score: {final_score}\n")
    
    logger.info("Results saved to optimization_results.txt")