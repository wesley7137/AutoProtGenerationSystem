import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import logging
from Phase_2.analyze import run_analysis_pipeline
from Phase_2.generate import generate_protein_sequence, save_results_to_csv
from Phase_2.optimize_new import run_optimization_pipeline
from Phase_2.predict import run_prediction_pipeline, create_3d_model_esm3
from Phase_2.simulate import run_simulation_pipeline
from datetime import datetime
from utils.save_utils import save_results_to_csv, save_partial_results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def predict_simulate_analyze(sequence, pred_dir, sim_dir, analysis_dir):
    try:
        logger.info(f"Running prediction pipeline...")
        prediction_results = await run_prediction_pipeline([sequence], output_dir=pred_dir)
        
        if not prediction_results or not prediction_results[0]['pdb_file']:
            logger.error(f"Prediction failed. Skipping simulation and analysis.")
            return None

        prediction_result = prediction_results[0]
        pdb_file = prediction_result['pdb_file']
        refined_sequence = prediction_result['refined_sequence']

        logger.info(f"Running simulation...")
        simulation_result = await run_simulation_pipeline(pdb_file, sim_dir)

        logger.info(f"Analyzing results...")
        analysis_result = await run_analysis_pipeline(
            simulation_result['trajectory_file'],
            simulation_result['final_pdb'],
            analysis_dir
        )

        # Generate 3D model using PyMOL
        model_file = os.path.join(analysis_dir, f"3d_model.png")
        await create_3d_model_esm3(pdb_file, f"sequence_{os.path.basename(os.path.dirname(analysis_dir))}", model_file)

        return {
            "sequence": refined_sequence,
            "pdb_file": pdb_file,
            "simulation_result": simulation_result,
            "analysis_result": analysis_result,
            "prediction_result": prediction_result
        }
    except Exception as e:
        logger.error(f"Error in predict_simulate_analyze: {str(e)}")
        return None    
    
    
    

async def run_Phase_2(input_text, optimization_steps, score_threshold, run_dir, results_dir, predicted_structures_dir, technical_descriptions):
    logger.info("\nStarting Phase 2: Generating and analyzing novel proteins")

    all_analysis_results = []
    best_score = 0  # Initialize the best score

    for i, desc in enumerate(technical_descriptions):
        technical_instruction = desc['technical_instruction']
        logger.info(f"\nProcessing Technical Description {i+1}:")
        logger.info(f"- {technical_instruction}")
        
        attempts = 0
        max_attempts = 10  # Set a maximum number of attempts to prevent infinite loops
        
        while attempts < max_attempts:
            # Generate protein sequence
            generated_sequences = await generate_protein_sequence(technical_instruction)
            
            if not generated_sequences:
                logger.warning(f"Failed to generate sequence for attempt {attempts + 1}. Retrying...")
                attempts += 1
                continue

            generated_sequence = generated_sequences[0]  # Take the first generated sequence
            logger.info(f"Generated sequence (attempt {attempts + 1}): {generated_sequence[:50]}...")

            # Optimize the generated sequence
            optimized_results = await run_optimization_pipeline([generated_sequence], iterations=optimization_steps, score_threshold=score_threshold)
            if optimized_results:
                logger.info(f"Optimized {len(optimized_results)} sequences")

                for opt_result in optimized_results:
                    optimized_sequence = opt_result['optimized_sequence']
                    original_score = opt_result['original_score']
                    optimized_score = opt_result['optimized_score']
                    best_method = opt_result['best_method']

                    logger.info(f"Optimized sequence (using {best_method}):")
                    logger.info(f"Original score: {original_score}, Optimized score: {optimized_score}")
                    logger.info(f"Sequence: {optimized_sequence[:50]}...")

            if optimized_score > best_score:
                # Create directories for this sequence
                analysis_dir = os.path.join(results_dir, f"structure_{len(all_analysis_results)}", "analysis")
                simulation_dir = os.path.join(results_dir, f"structure_{len(all_analysis_results)}", "simulation")
                os.makedirs(analysis_dir, exist_ok=True)
                os.makedirs(simulation_dir, exist_ok=True)

                # Run prediction, simulation, and analysis on the optimized sequence
                result = await predict_simulate_analyze(optimized_sequence, predicted_structures_dir, simulation_dir, analysis_dir)
                if result:
                    result['optimization_info'] = {
                            'original_score': original_score,
                            'optimized_score': optimized_score,
                            'best_method': best_method
                        }
                    result['technical_description'] = technical_instruction
                    all_analysis_results.append(result)
                    best_score = optimized_score
                    logger.info(f"New best score: {best_score}")
                    await save_partial_results(all_analysis_results, run_dir, i)
                    break  # Exit the while loop as we found a better sequence
                else:
                    logger.info(f"Skipping simulation for sequence with score {optimized_score} (not better than {best_score})")
            
            attempts += 1
            if attempts == max_attempts:
                logger.info(f"Reached maximum attempts ({max_attempts}) for Technical Description {i+1}. Moving to next description.")

    # Save final results
    if all_analysis_results:
        final_sequences = [result['sequence'] for result in all_analysis_results]
        final_scores = [result['analysis_result'].get('final_score', 0) for result in all_analysis_results]
        optimization_scores = [result['optimization_info']['optimized_score'] for result in all_analysis_results]
        optimization_methods = [result['optimization_info']['best_method'] for result in all_analysis_results]
        technical_descriptions = [result['technical_description'] for result in all_analysis_results]
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(run_dir, f'overall_results_{start_time}.csv')
        await save_results_to_csv(
            input_text, 
            final_sequences, 
            final_scores, 
            start_time, 
            results_file, 
            optimization_scores=optimization_scores,
            optimization_methods=optimization_methods,
            technical_descriptions=technical_descriptions
        )
        logger.info(f"Final results saved to: {results_file}")
    else:
        logger.info("No successful results to save.")

    return all_analysis_results