import os
import asyncio
import logging
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, DSSP
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from pymol import cmd

# Import your existing functions
from Phase_2.analyze import run_analysis_pipeline
from Phase_2.generate import generate_protein_sequence, save_results_to_csv
from Phase_2.optimize4 import run_optimization_pipeline
from Phase_2.predict import run_prediction_pipeline, predict_protein_function, predict_properties, predict_structure
from Phase_2.simulate import run_simulation_pipeline
from Phase_2.generate import read_technical_descriptions
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the ESM3 model
esm3_model = ESM3InferenceClient.from_pretrained("esm3_sm_open_v1").to("cuda")  # or "cpu"

def create_3d_model_esm3(protein_pdb, prompt_string, output_dir):
    """Create a 3D model visualization using PyMOL."""
    pymol.finish_launching(['pymol', '-cq'])  # '-cq' for command line only and quiet mode
    
    try:
        # Load the PDB file into PyMOL
        cmd.load(protein_pdb)
        
        # Apply color and visualization settings
        cmd.spectrum("count", "rainbow", "all")
        cmd.show("cartoon")
        cmd.bg_color("black")
        
        # Save the image
        image_filename = os.path.join(output_dir, f"{prompt_string}_model.png")
        cmd.png(image_filename)
        logger.info(f"Image saved as {image_filename}")
        
        # Save the PyMOL session if needed
        session_filename = os.path.join(output_dir, f"{prompt_string}_model.pse")
        cmd.save(session_filename)
        logger.info(f"PyMOL session saved as {session_filename}")
        
    except pymol.CmdException as e:
        logger.error(f"PyMOL command failed - {e}")
    finally:
        cmd.quit()

async def esm3_refinement(sequence):
    """Refine the sequence using ESM3 model predictions."""
    try:
        protein = ESMProtein(sequence=sequence)
        # Refine sequence using ESM3 model
        refined_protein = esm3_model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
        refined_protein = esm3_model.generate(refined_protein, GenerationConfig(track="structure", num_steps=8))
        logger.info(f"Refined sequence: {refined_protein.sequence[:50]}...")  # Log first 50 characters
        return refined_protein.sequence
    except Exception as e:
        logger.error(f"Error in ESM3 refinement: {str(e)}")
        return sequence  # Return original if refinement fails

async def predict_simulate_analyze(sequence, output_dir, sequence_id):
    # Refine the sequence with ESM3 model
    refined_sequence = await esm3_refinement(sequence)
    
    logger.info(f"Predicting structure for sequence {sequence_id}...")
    structure_output_dir = os.path.join(output_dir, f"structure_{sequence_id}")
    pdb_file = await predict_structure(refined_sequence, structure_output_dir)
    
    if not pdb_file:
        logger.error(f"Prediction failed for sequence {sequence_id}. Skipping simulation and analysis.")
        return None

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
    create_3d_model_esm3(pdb_file, f"sequence_{sequence_id}", analysis_output_dir)

    return {
        "sequence": refined_sequence,
        "pdb_file": pdb_file,
        "simulation_result": simulation_result,
        "analysis_result": analysis_result
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
        queue = asyncio.Queue()
        optimized_sequences = []

        # Start generator and optimizer tasks
        generator_task = asyncio.create_task(generate_sequences(input_text, num_sequences, queue))
        optimizer_task = asyncio.create_task(optimize_sequences(optimization_steps, score_threshold, queue, optimized_sequences))

        # Wait for both generator and optimizer to complete
        await asyncio.gather(generator_task, optimizer_task)

        # Run full pipeline on optimized sequences
        if optimized_sequences:
            logger.info(f"Optimized {len(optimized_sequences)} sequences")
            analysis_results = await run_full_pipeline(optimized_sequences, output_dir)

            # Save results
            final_sequences = [result['sequence'] for result in analysis_results]
            final_scores = [result['analysis_result'].get('final_score', 0) for result in analysis_results]
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = await save_results_to_csv(input_text, final_sequences, final_scores, start_time, output_dir)
            logger.info(f"Final results saved to: {results_file}")
        else:
            logger.info("No sequences were optimized successfully.")

    return "Phase 2 completed successfully"
