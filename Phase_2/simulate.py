import os
import asyncio
import logging
import mdtraj as md

from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer
import nglview as nv
import mdtraj as md

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def prepare_system(pdb_file):
    """Asynchronously prepare the system for simulation."""
    try:
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * nanometers)

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
                                         nonbondedCutoff=1.0 * nanometers,
                                         constraints=HBonds)
        logger.info(f"System prepared for: {pdb_file}")
        return system, modeller
    except Exception as e:
        logger.error(f"Error preparing system for {pdb_file}: {str(e)}")
        raise



async def run_simulation(system, modeller, output_dir, steps=10000):
    """Asynchronously run the molecular dynamics simulation."""
    try:
        integrator = LangevinMiddleIntegrator(300*kelvin, 1.0/picosecond, 2.0*femtosecond)
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}

        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)

        logger.info("Minimizing energy...")
        simulation.minimizeEnergy()

        logger.info("Equilibrating...")
        simulation.context.setVelocitiesToTemperature(300*kelvin)
        simulation.step(10000)  # 20 picoseconds of equilibration (10000 * 2 fs)

        logger.info(f"Running production simulation for {steps} steps...")
        simulation.reporters.append(DCDReporter(os.path.join(output_dir, 'trajectory.dcd'), 1000))
        simulation.reporters.append(StateDataReporter(os.path.join(output_dir, 'output.txt'), 1000, 
            step=True, potentialEnergy=True, temperature=True, progress=True, 
            remainingTime=True, speed=True, totalSteps=steps, separator='\t'))

        # Run the simulation with progress logging
        for i in range(0, steps, 1000):
            simulation.step(1000)
            logger.info(f"Simulation progress: {i+1000}/{steps} steps ({(i+1000)/500:.1f} ps)")

        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open(os.path.join(output_dir, 'final.pdb'), 'w'))

        logger.info("Simulation completed successfully.")
        return os.path.join(output_dir, 'trajectory.dcd'), os.path.join(output_dir, 'final.pdb')
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise
    
    
async def visualize_trajectory(trajectory_file, pdb_file, output_file):
    """Create an HTML visualization of the trajectory."""
    try:
        traj = md.load(trajectory_file, top=pdb_file)
        view = nv.show_mdtraj(traj)
        view.render_image()
        view.download_image(output_file)
        logger.info(f"Trajectory visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating trajectory visualization: {str(e)}")
        logger.info("Skipping visualization due to error.")
        return None  # Return None instead of raising an exception
    
    
async def run_simulation_pipeline(pdb_file, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Preparing system for {pdb_file}")
        system, modeller = await prepare_system(pdb_file)

        logger.info(f"Running simulation for {pdb_file}")
        trajectory_file, final_pdb = await run_simulation(system, modeller, output_dir)

        # Create visualization
        visualization_file = os.path.join(output_dir, 'trajectory_visualization.png')
        await visualize_trajectory(trajectory_file, final_pdb, visualization_file)

        logger.info(f"Simulation results saved in: {output_dir}")

        return {
            "input_pdb": pdb_file,
            "output_dir": output_dir,
            "trajectory_file": trajectory_file,
            "final_pdb": final_pdb,
            "visualization_file": visualization_file
        }
    except Exception as e:
        logger.error(f"Error in simulation pipeline for {pdb_file}: {str(e)}")
        raise