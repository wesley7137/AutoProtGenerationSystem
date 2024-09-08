import os
import asyncio
import logging
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, DSSP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def load_trajectory(trajectory_file, topology_file):
    """Asynchronously load the trajectory file."""
    try:
        return await asyncio.to_thread(md.load, trajectory_file, top=topology_file)
    except Exception as e:
        logger.error(f"Error loading trajectory: {str(e)}")
        raise

async def calculate_rmsd(trajectory):
    """Asynchronously calculate RMSD of the trajectory."""
    try:
        return await asyncio.to_thread(md.rmsd, trajectory, trajectory, 0)
    except Exception as e:
        logger.error(f"Error calculating RMSD: {str(e)}")
        raise

async def calculate_rmsf(trajectory):
    """Asynchronously calculate RMSF of the trajectory."""
    try:
        return await asyncio.to_thread(md.rmsf, trajectory, trajectory.topology.select("name CA"))
    except Exception as e:
        logger.error(f"Error calculating RMSF: {str(e)}")
        raise

async def calculate_radius_of_gyration(trajectory):
    """Asynchronously calculate radius of gyration of the trajectory."""
    try:
        return await asyncio.to_thread(md.compute_rg, trajectory)
    except Exception as e:
        logger.error(f"Error calculating radius of gyration: {str(e)}")
        raise

async def calculate_secondary_structure(pdb_file):
    """Asynchronously calculate secondary structure composition."""
    try:
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file, dssp='mkdssp')

        ss_counts = {'H': 0, 'E': 0, 'C': 0}
        for residue in dssp:
            ss = residue[2]
            if ss in ['H', 'G', 'I']:
                ss_counts['H'] += 1  # Helix
            elif ss in ['E', 'B']:
                ss_counts['E'] += 1  # Sheet
            else:
                ss_counts['C'] += 1  # Coil

        total = sum(ss_counts.values())
        ss_composition = {k: v / total for k, v in ss_counts.items()}
        return ss_composition
    except Exception as e:
        logger.error(f"Error calculating secondary structure: {str(e)}")
        raise

async def plot_data(data, title, xlabel, ylabel, output_file):
    """Asynchronously plot data."""
    try:
        await asyncio.to_thread(plt.figure, figsize=(10, 6))
        await asyncio.to_thread(plt.plot, data)
        await asyncio.to_thread(plt.title, title)
        await asyncio.to_thread(plt.xlabel, xlabel)
        await asyncio.to_thread(plt.ylabel, ylabel)
        await asyncio.to_thread(plt.savefig, output_file)
        await asyncio.to_thread(plt.close)
        logger.info(f"{title} plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Error plotting {title}: {str(e)}")
        raise

async def run_analysis_pipeline(trajectory_file, topology_file, final_pdb, output_dir):
    """Asynchronously run the full analysis pipeline."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Loading trajectory...")
        traj = await load_trajectory(trajectory_file, topology_file)

        logger.info("Calculating RMSD...")
        rmsd = await calculate_rmsd(traj)
        await plot_data(rmsd, 'RMSD over time', 'Frame', 'RMSD (nm)', os.path.join(output_dir, 'rmsd_plot.png'))

        logger.info("Calculating RMSF...")
        rmsf = await calculate_rmsf(traj)
        await plot_data(rmsf, 'RMSF per residue', 'Residue', 'RMSF (nm)', os.path.join(output_dir, 'rmsf_plot.png'))

        logger.info("Calculating radius of gyration...")
        rg = await calculate_radius_of_gyration(traj)
        await plot_data(rg, 'Radius of Gyration over time', 'Frame', 'Rg (nm)', os.path.join(output_dir, 'rg_plot.png'))

        logger.info("Analyzing secondary structure...")
        ss_composition = await calculate_secondary_structure(final_pdb)

        analysis_results = {
            "rmsd": rmsd.tolist(),
            "rmsf": rmsf.tolist(),
            "radius_of_gyration": rg.tolist(),
            "secondary_structure": ss_composition
        }

        logger.info("Analysis pipeline completed successfully.")
        return analysis_results
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise
