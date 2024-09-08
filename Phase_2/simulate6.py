import os
import asyncio
import logging
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer

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

async def run_simulation(system, modeller, output_dir, steps=500000):
    """Asynchronously run the molecular dynamics simulation."""
    try:
        integrator = LangevinMiddleIntegrator(300 * kelvin, 1.0 / picoseconds, 2.0 * femtoseconds)
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}

        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)

        logger.info("Minimizing energy...")
        simulation.minimizeEnergy()

        logger.info("Equilibrating...")
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
        simulation.step(10000)

        logger.info(f"Running production simulation for {steps} steps...")
        simulation.reporters.append(DCDReporter(os.path.join(output_dir, 'trajectory.dcd'), 1000))
        simulation.reporters.append(StateDataReporter(os.path.join(output_dir, 'output.txt'), 1000, 
            step=True, potentialEnergy=True, temperature=True, progress=True, 
            remainingTime=True, speed=True, totalSteps=steps, separator='\t'))

        simulation.step(steps)

        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open(os.path.join(output_dir, 'final.pdb'), 'w'))

        logger.info("Simulation completed successfully.")
        return os.path.join(output_dir, 'trajectory.dcd'), os.path.join(output_dir, 'final.pdb')
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise

async def run_simulation_pipeline(pdb_file, output_dir):
    """Asynchronously run the full simulation pipeline for a given PDB file."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Preparing system for {pdb_file}")
        system, modeller = await prepare_system(pdb_file)

        logger.info(f"Running simulation for {pdb_file}")
        trajectory_file, final_pdb = await run_simulation(system, modeller, output_dir)

        return {
            "input_pdb": pdb_file,
            "output_dir": output_dir,
            "trajectory_file": trajectory_file,
            "final_pdb": final_pdb
        }
    except Exception as e:
        logger.error(f"Error in simulation pipeline for {pdb_file}: {str(e)}")
        raise
