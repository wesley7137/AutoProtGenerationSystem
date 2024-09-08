
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import platform
import asyncio
import aioconsole

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
import pymol
import os
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import truncnorm
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm3_model:ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")  # or "cpu"


async def predict_structure(sequence):
    """Predict the structure of a protein sequence using ESM-3."""
    try:
        protein = ESMProtein(sequence=sequence)
        config = GenerationConfig(track="structure", num_steps=8)
        result = esm3_model.generate(protein, config)
        
        logger.info(f"Structure generation result type: {type(result)}")
        logger.info(f"Structure generation result attributes: {dir(result)}")
        
        # Save the PDB file
        output_dir = "predicted_structures"
        os.makedirs(output_dir, exist_ok=True)
        pdb_file = os.path.join(output_dir, f"{sequence[:10]}_structure.pdb")
        result.to_pdb(pdb_file)
        
        logger.info(f"Structure predicted and saved to {pdb_file}")
        return pdb_file
    except Exception as e:
        logger.error(f"Error in predict_structure: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error args: {e.args}")
        return None


async def predict_protein_function(sequence):
    """Asynchronously predict the function score of a given protein sequence."""
    try:
        print(f"Received sequence for prediction: {sequence}")
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence = ''.join(char.upper() for char in sequence if char.upper() in valid_aa)
        print(f"Cleaned sequence: {sequence}")
        if not sequence:
            print("Empty or invalid sequence for protein function prediction")
            logger.error("Empty or invalid sequence for protein function prediction")
            return 0.5

        analysis = ProteinAnalysis(sequence)
        molecular_weight = analysis.molecular_weight()
        aromaticity = analysis.aromaticity()
        instability_index = analysis.instability_index()
        isoelectric_point = analysis.isoelectric_point()

        norm_weight = truncnorm.cdf((molecular_weight - 25000) / 10000, -2, 2)
        norm_aromaticity = aromaticity
        norm_instability = 1 - truncnorm.cdf((instability_index - 40) / 10, -2, 2)
        norm_isoelectric = truncnorm.cdf((isoelectric_point - 7) / 2, -2, 2)
        aa_count = {aa: sequence.count(aa) for aa in valid_aa}
        total_aa = len(sequence)
        composition_balance = 1 - sum(abs(count/total_aa - 0.05) for count in aa_count.values()) / 2

        weights = [0.25, 0.15, 0.25, 0.15, 0.2]
        score = sum(w * v for w, v in zip(weights, [norm_weight, norm_aromaticity, norm_instability, norm_isoelectric, composition_balance]))
        print(f"Predicted function score for sequence: {score}")
        logger.info(f"Predicted function score for sequence: {score}")
        return max(0, min(1, score))
    except Exception as e:
        print(f"Error in predict_protein_function: {str(e)}")
        logger.error(f"Error in predict_protein_function: {str(e)}")
        return 0.5
    
    
    
    
async def predict_properties(sequence):
    """Asynchronously predict various properties of a protein sequence."""
    try:
        analysis = ProteinAnalysis(sequence)
        properties = {
            "molecular_weight": analysis.molecular_weight(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy(),
            "secondary_structure_fraction": analysis.secondary_structure_fraction()
        }
        logger.info(f"Predicted properties for sequence: {properties}")
        return properties
    except Exception as e:
        logger.error(f"Error in predict_properties: {str(e)}")
        return {}



async def create_3d_model_esm3(protein_pdb, prompt_string, output_dir):
    """Create a 3D model visualization using PyMOL and provide instructions to open it."""
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
        
        # Save the PyMOL session
        session_filename = os.path.join(output_dir, f"{prompt_string}_model.pse")
        cmd.save(session_filename)
        logger.info(f"PyMOL session saved as {session_filename}")
        
        # Close PyMOL
        cmd.quit()
        
        # Provide instructions to open PyMOL
        print("\nTo view the 3D structure:")
        print(f"1. Open PyMOL")
        print(f"2. In PyMOL, go to File -> Open")
        print(f"3. Navigate to and select this file: {os.path.abspath(session_filename)}")
        
        try:
            user_input = await asyncio.wait_for(
                aioconsole.ainput("Press Enter when you're ready to continue, or type 'skip' to move on: "),
                timeout=30.0
            )
            if user_input.lower() == 'skip':
                logger.info("User chose to skip opening PyMOL.")
            else:
                logger.info("User acknowledged PyMOL instructions.")
        except asyncio.TimeoutError:
            logger.info("No input received within 30 seconds. Continuing.")
        
    except pymol.CmdException as e:
        logger.error(f"PyMOL command failed - {e}")





async def esm3_refinement(sequence):
    """Refine the sequence using ESM3 model predictions."""
    try:
        protein = ESMProtein(sequence=sequence)
        # Refine sequence using ESM3 model
        refined_protein = esm3_model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
        refined_sequence = refined_protein.sequence
        logger.info(f"Refined sequence: {refined_sequence[:50]}...")  # Log first 50 characters
        return refined_sequence
    except Exception as e:
        logger.error(f"Error in ESM3 refinement: {str(e)}")
        return sequence  # Return original if refinement fails


    
async def run_prediction_pipeline(sequences):
    """Asynchronously run the full prediction pipeline for a list of sequences."""
    tasks = []
    for i, sequence in enumerate(sequences):
        if isinstance(sequence, dict):
            sequence = sequence['optimized_sequence']
        logger.info(f"Processing sequence {i+1}/{len(sequences)}: {sequence}")
        refined_sequence = await esm3_refinement(sequence)
        tasks.append(asyncio.gather(
            predict_protein_function(refined_sequence),
            predict_properties(refined_sequence),
            predict_structure(refined_sequence)
        ))
    
    results = await asyncio.gather(*tasks)
    return [
        {
            "sequence": seq,
            "refined_sequence": refined_seq,
            "score": res[0],
            "properties": res[1],
            "pdb_file": res[2]
        } for seq, refined_seq, res in zip(sequences, [await esm3_refinement(s) for s in sequences], results)
    ]
    
    
async def main():
    # Test sequence
    test_sequence = "ITSPGCSYISLDVTPANLAAVNNRFTIASVNARYRALLAADRDFVEQYALRFYDKTRHHYTIERATQPDGLALTQFFIDQPTNQGEYTRPNAQNFQITQDFTYYPQKAKLRSGLGVVTVYDLSPINQSLGKPPANFTVLSHVFEHLLAGSNYHRVSFELLTTGYTISAAVHRGGTLPAIAKELTRQDDSPTLTGQLATRARELKEYCFATQINLRGNSGSRGNPPCPNESSAFIKFAPAPISNFTQQIQGAANEL"
    
    print("Running prediction pipeline...")
    results = await run_prediction_pipeline([test_sequence])
    
    for result in results:
        print("\nPrediction Results:")
        print(f"Original Sequence: {result['sequence'][:50]}...")
        print(f"Refined Sequence: {result['refined_sequence'][:50]}...")
        print(f"Function Score: {result['score']}")
        print("Properties:")
        for prop, value in result['properties'].items():
            print(f"  {prop}: {value}")
        print(f"PDB file: {result['pdb_file']}")
        
        if result['pdb_file']:
            output_dir = "predicted_structures"
            await create_3d_model_esm3(result['pdb_file'], "test_protein", output_dir)

if __name__ == "__main__":
    asyncio.run(main())