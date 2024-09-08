import os
import asyncio
import logging
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from Bio import PDB

logger = logging.getLogger(__name__)

import mdtraj as md
import numpy as np
from Bio import PDB

import mdtraj as md
import numpy as np
import logging

logger = logging.getLogger(__name__)

async def load_trajectory(trajectory_file, topology_file):
    """Load the trajectory asynchronously."""
    logger.info("Loading trajectory...")
    try:
        # Load the topology file using MDTraj
        topology = md.load_pdb(topology_file).topology

        # Load the trajectory using the topology
        traj = md.load(trajectory_file, top=topology)
        
        logger.info(f"Trajectory loaded successfully. Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
        return traj
    except Exception as e:
        logger.error(f"Error loading trajectory: {str(e)}")
        # Print more details about the files
        with open(topology_file, 'r') as f:
            logger.error(f"First few lines of topology file:\n{f.read(500)}")
        logger.error(f"Trajectory file size: {os.path.getsize(trajectory_file)} bytes")
        raise
    
    
    
async def calculate_rmsd(traj):
    """Calculate RMSD of the trajectory."""
    logger.info("Calculating RMSD...")
    return np.sqrt(3*np.mean(np.sum(np.square(traj.xyz - traj.xyz.mean(axis=0)), axis=2), axis=1))

async def calculate_rmsf(traj):
    """Calculate RMSF of the trajectory."""
    logger.info("Calculating RMSF...")
    return np.sqrt(3*np.mean(np.square(traj.xyz - np.mean(traj.xyz, axis=0)), axis=0))

async def calculate_radius_of_gyration(traj):
    """Calculate radius of gyration of the trajectory."""
    logger.info("Calculating radius of gyration...")
    return md.compute_rg(traj)

async def calculate_secondary_structure(traj):
    """Calculate secondary structure of the trajectory."""
    logger.info("Calculating secondary structure...")
    ss = md.compute_dssp(traj)
    
    # Convert string labels to numerical values
    ss_map = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
    ss_numeric = np.array([[ss_map.get(s, 2) for s in frame] for frame in ss])
    
    return ss_numeric

async def generate_rmsd_plot(rmsd, output_dir):
    """Generate RMSD plot."""
    plt.figure()
    plt.plot(rmsd)
    plt.xlabel('Frame')
    plt.ylabel('RMSD (nm)')
    plt.savefig(os.path.join(output_dir, 'rmsd_plot.png'))
    plt.close()

async def generate_rmsf_plot(rmsf, output_dir):
    """Generate RMSF plot."""
    plt.figure()
    plt.plot(rmsf)
    plt.xlabel('Residue')
    plt.ylabel('RMSF (nm)')
    plt.savefig(os.path.join(output_dir, 'rmsf_plot.png'))
    plt.close()

async def generate_rg_plot(rg, output_dir):
    """Generate radius of gyration plot."""
    plt.figure()
    plt.plot(rg)
    plt.xlabel('Frame')
    plt.ylabel('Radius of Gyration (nm)')
    plt.savefig(os.path.join(output_dir, 'rg_plot.png'))
    plt.close()

async def generate_ss_plot(ss, output_dir):
    """Generate secondary structure plot."""
    plt.figure(figsize=(10, 6))
    plt.imshow(ss.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xlabel('Frame')
    plt.ylabel('Residue')
    plt.colorbar(label='Secondary Structure')
    plt.title('Secondary Structure Evolution')
    
    # Add custom colorbar ticks
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Helix', 'Sheet', 'Coil'])
    
    plt.savefig(os.path.join(output_dir, 'ss_plot.png'))
    plt.close()

import numpy as np

async def calculate_final_score(rmsd, rmsf, rg, ss):
    """Calculate a final score based on the analysis results."""
    try:
        # 1. RMSD stability score (lower is better)
        rmsd_mean = np.mean(rmsd)
        rmsd_std = np.std(rmsd)
        rmsd_score = 1 / (1 + rmsd_mean + rmsd_std)  # Normalized between 0 and 1

        # 2. RMSF flexibility score (balance between flexibility and rigidity)
        rmsf_mean = np.mean(rmsf)
        rmsf_score = np.exp(-(rmsf_mean - 0.1)**2 / 0.02)  # Gaussian centered at 0.1 nm

        # 3. Radius of gyration compactness score (prefer compact structures)
        rg_mean = np.mean(rg)
        rg_score = 1 / (1 + rg_mean)  # Normalized between 0 and 1

        # 4. Secondary structure stability score
        ss_counts = np.sum(ss, axis=0)
        helix_percent = ss_counts[0] / np.sum(ss_counts)
        sheet_percent = ss_counts[1] / np.sum(ss_counts)
        ss_score = (helix_percent + sheet_percent) / 2  # Prefer more structured elements

        # Calculate final score (weighted average)
        weights = [0.3, 0.2, 0.2, 0.3]  # Adjust these weights as needed
        final_score = np.dot([rmsd_score, rmsf_score, rg_score, ss_score], weights)

        # Normalize final score between 0 and 1
        final_score = (final_score - 0.5) * 2  # Assuming 0.5 is an average score

        return max(0, min(1, final_score))  # Clamp between 0 and 1

    except Exception as e:
        logger.error(f"Error calculating final score: {str(e)}")
        return 0  # Return 0 if there's an error
    
    

async def run_analysis_pipeline(trajectory_file, final_pdb, output_dir):
    """Run the full analysis pipeline on the simulation results."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting analysis pipeline with trajectory file: {trajectory_file}")
        logger.info(f"Using topology file: {final_pdb}")
        logger.info(f"Output directory: {output_dir}")

        # Use the final PDB as the topology file
        traj = await load_trajectory(trajectory_file, final_pdb)

        if traj is None or traj.n_frames == 0:
            logger.error("Failed to load trajectory or trajectory is empty.")
            return None

        # Perform various analyses
        rmsd = await calculate_rmsd(traj)
        rmsf = await calculate_rmsf(traj)
        radius_of_gyration = await calculate_radius_of_gyration(traj)
        secondary_structure = await calculate_secondary_structure(traj)

        # Generate plots
        await generate_rmsd_plot(rmsd, output_dir)
        await generate_rmsf_plot(rmsf, output_dir)
        await generate_rg_plot(radius_of_gyration, output_dir)
        await generate_ss_plot(secondary_structure, output_dir)

        # Calculate a final score based on the analysis results
        final_score = await calculate_final_score(rmsd, rmsf, radius_of_gyration, secondary_structure)

        logger.info(f"Analysis completed. Final score: {final_score}")

        return {
            "rmsd": rmsd.tolist(),
            "rmsf": rmsf.tolist(),
            "radius_of_gyration": radius_of_gyration.tolist(),
            "secondary_structure": secondary_structure.tolist(),
            "final_score": final_score
        }

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        logger.error(f"Trajectory file: {trajectory_file}")
        logger.error(f"Topology file: {final_pdb}")
        logger.error(f"Output directory: {output_dir}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None