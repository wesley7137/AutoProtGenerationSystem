import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import asyncio
import logging
# Import your existing functions
from Phase_2.analyze7 import run_analysis_pipeline
from Phase_2.generate3 import generate_protein_sequence, save_results_to_csv
from Phase_2.optimize4 import run_optimization_pipeline
from Phase_2.predict5 import run_prediction_pipeline, create_3d_model_esm3
from Phase_2.simulate6 import run_simulation_pipeline
from Phase_2.generate3 import read_technical_descriptions
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the ESM3 model

async def predict_simulate_analyze(sequence, output_dir, sequence_id):
    logger.info(f"Running prediction pipeline for sequence {sequence_id}...")
    prediction_results = await run_prediction_pipeline([sequence])
    
    if not prediction_results or not prediction_results[0]['pdb_file']:
        logger.error(f"Prediction failed for sequence {sequence_id}. Skipping simulation and analysis.")
        return None

    prediction_result = prediction_results[0]
    pdb_file = prediction_result['pdb_file']
    refined_sequence = prediction_result['refined_sequence']

    logger.info(f"Running simulation for sequence {sequence_id}...")
    simulation_output_dir = os.path.join(output_dir, f"simulation_{sequence_id}")
    simulation_result = await run_simulation_pipeline(pdb_file, simulation_output_dir)

    logger.info(f"Analyzing results for sequence {sequence_id}...")
    analysis_output_dir = os.path.join(output_dir, f"analysis_{sequence_id}")
    analysis_result = await run_analysis_pipeline(
        simulation_result['trajectory_file'],
        pdb_file,
        simulation_result['final_pdb'],
        analysis_output_dir
    )

    # Generate 3D model using PyMOL
    await create_3d_model_esm3(pdb_file, f"sequence_{sequence_id}", analysis_output_dir)

    return {
        "sequence": refined_sequence,
        "pdb_file": pdb_file,
        "simulation_result": simulation_result,
        "analysis_result": analysis_result,
        "prediction_result": prediction_result
    }

async def run_full_pipeline(optimized_sequences, output_dir):
    tasks = [
        predict_simulate_analyze(sequence, output_dir, i) 
        for i, sequence in enumerate(optimized_sequences)
    ]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result]

async def run_Phase_2(base_dir, input_text, num_sequences, optimization_steps, score_threshold, output_dir):
    technical_descriptions = read_technical_descriptions(base_dir)

    print("\nProcessing Technical Descriptions:")
    for desc in technical_descriptions:
        technical_instruction = desc['technical_instruction']
        print(f"- {technical_instruction}")
        
        # Generate protein sequence
        generated_sequence = await generate_protein_sequence(input_text)
        logger.info(f"Generated sequence: {generated_sequence[:50]}...")  # Log first 50 characters

        # Optimize the generated sequence
        optimized_sequences = await run_optimization_pipeline([generated_sequence], iterations=optimization_steps, score_threshold=score_threshold)

        # Run full pipeline on optimized sequences
        if optimized_sequences:
            logger.info(f"Optimized {len(optimized_sequences)} sequences")
            # Extract only the optimized sequences from the optimization results
            sequences_for_prediction = [seq['optimized_sequence'] for seq in optimized_sequences]
            analysis_results = await run_full_pipeline(sequences_for_prediction, output_dir)

            # Save results
            final_sequences = [result['sequence'] for result in analysis_results if result]
            final_scores = [result['analysis_result'].get('final_score', 0) for result in analysis_results if result]
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = await save_results_to_csv(input_text, final_sequences, final_scores, start_time, output_dir)
            logger.info(f"Final results saved to: {results_file}")
        else:
            logger.info("No sequences were optimized successfully.")

    return "Phase 2 completed successfully"