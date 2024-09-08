import os
from datetime import datetime
import csv
import asyncio

def create_organized_directory_structure(base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    
    # Create subdirectories
    predicted_structures_dir = os.path.join(run_dir, "predicted_structures")
    results_dir = os.path.join(run_dir, "results")
    
    os.makedirs(predicted_structures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return run_dir, predicted_structures_dir, results_dir

def create_sequence_directories(results_dir, sequence_id):
    analysis_dir = os.path.join(results_dir, f"structure_{sequence_id}", "analysis")
    simulation_dir = os.path.join(results_dir, f"structure_{sequence_id}", "simulation")
    
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)
    
    return analysis_dir, simulation_dir

async def save_results_to_csv(input_text, sequences, scores, timestamp, filename, **kwargs):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input Text', 'Sequence', 'Score', 'Timestamp'] + list(kwargs.keys()))
        for i, (seq, score) in enumerate(zip(sequences, scores)):
            row = [input_text, seq, score, timestamp]
            row.extend([kwargs[key][i] for key in kwargs])
            writer.writerow(row)

async def save_partial_results(results, run_dir, description_index):
    if results:
        partial_results_file = os.path.join(run_dir, f'partial_results_description_{description_index}.csv')
        final_sequences = [result['sequence'] for result in results]
        final_scores = [result['analysis_result'].get('final_score', 0) for result in results]
        optimization_scores = [result['optimization_info']['optimized_score'] for result in results]
        optimization_methods = [result['optimization_info']['best_method'] for result in results]
        technical_descriptions = [result['technical_description'] for result in results]
        
        await save_results_to_csv(
            f"Partial results for description {description_index}", 
            final_sequences, 
            final_scores, 
            datetime.now().strftime("%Y%m%d_%H%M%S"), 
            partial_results_file, 
            optimization_scores=optimization_scores,
            optimization_methods=optimization_methods,
            technical_descriptions=technical_descriptions
        )
        return partial_results_file
    return None