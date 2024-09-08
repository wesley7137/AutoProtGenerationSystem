# Protein Structure and Interaction Pipeline

## Overview

This project implements an automated pipeline for the generation, optimization, and analysis of protein structures using state-of-the-art AI models and molecular simulation tools. The goal is to facilitate the discovery of novel proteins and interactions relevant to longevity research and other biotechnological applications. The pipeline integrates ProGen for protein sequence generation, AlphaFold for structure prediction, OpenMM for molecular dynamics simulations, PyRosetta for mutagenesis and optimization, HADDOCK for docking simulations, and PyMOL for visualization.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automated Protein Sequence Generation**: Uses ProGen to generate protein sequences based on specified functions or descriptions.
- **Structure Prediction**: Predicts 3D structures of generated sequences using AlphaFold.
- **Molecular Dynamics Simulation**: Refines the predicted structures using OpenMM to simulate molecular dynamics and energy minimization.
- **Mutagenesis and Optimization**: Employs PyRosetta to perform mutagenesis and structure optimization through automated relaxation protocols.
- **Docking Simulations**: Utilizes HADDOCK to simulate protein-protein and protein-ligand interactions.
- **Visualization**: Visualizes the final optimized structures using PyMOL for detailed structural analysis.

## Prerequisites

Ensure the following software and dependencies are installed and configured:

- Python 3.8 or later
- CUDA (if using GPU acceleration)
- ProGen: Installed via progeny package
- AlphaFold: Requires AlphaFold dependencies and model weights
- OpenMM: Installed for molecular dynamics simulations
- PyRosetta: Installed for protein structure refinement and mutagenesis
- HADDOCK: Properly installed and configured for docking simulations
- PyMOL: For visualizing protein structures
- inotify-tools: If using file monitoring on Linux (optional)
- SSH Key-based Authentication: For secure and automated data transfers between systems

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/protein-structure-pipeline.git
   cd protein-structure-pipeline
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all external dependencies (AlphaFold, OpenMM, PyRosetta, HADDOCK) are installed according to their respective installation guides.

4. Download the AlphaFold model weights and place them in the specified directory.

```bash
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
```

Extract the files:

```bash
tar -xvf alphafold_params_2022-12-06.tar
```

## Usage

1. Run the Pipeline: Execute the main script to start the full pipeline:

   ```bash
   python main.py
   ```

2. Monitor Directory for Changes (Optional): Use the provided PowerShell script or inotifywait-based script to automatically trigger rsync transfers when files are saved in the designated directories.

## Pipeline Steps

1. **Generate Protein Sequence with ProGen**:

   - Generates protein sequences based on provided descriptions using ProGen's pretrained models.

2. **Predict Protein Structure with ESM3**:

   - Uses ESM3 to predict the 3D structure of the generated sequences, saving the output in PDB format.

3. **Run Molecular Dynamics Simulation with OpenMM**:

   - Loads the predicted structures and performs energy minimization and dynamics simulation to refine the protein models.

4. **Automate Mutagenesis and Screening with PyRosetta**:

   - Applies mutagenesis and automated relaxation to optimize the protein structures for stability and function.

5. **Perform Docking Simulations with HADDOCK**:

   - Simulates docking of the optimized proteins with potential ligands or other proteins to explore interactions and binding affinities.

6. **Visualize Final Structures with PyMOL**:
   - Generates visual representations of the final optimized structures, providing insight into their geometrical and functional properties.

## Configuration

- **AlphaFold Model Path**: Ensure the AlphaFold model weights and configuration files are correctly set in the script (`alphafold_model_path`).
- **HADDOCK Configuration**: Verify that `dock.cfg` and any other necessary configuration files for HADDOCK are in place and properly configured.
- **Directory Paths**: Update the script with the correct paths for your local and remote directories for rsync operations.

## Troubleshooting

- **Permission Issues**: Ensure all scripts have the necessary execution permissions and that SSH keys are correctly set up for secure access between machines.
- **Dependency Errors**: Verify that all dependencies are installed and compatible with your system's architecture and Python version.
- **Resource Limitations**: Adjust resource allocations or run individual steps in isolation if encountering performance issues due to hardware limitations.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

#Phase 1
Step1: Generate novel hypothesis chemical and molecule ideas based on the rag based molecular database
Step 2: Check Idea Novelty
Step 3: Iterate. Hypothesis output from the previous step is the input for the next step

#Phase 2
Step 1: Generate a large number of protein sequences using ProGen
Step 2: Optimize the generated sequences using AlphaFold
Step 3: Refine the optimized structures using OpenMM
Step 4: Mutate the refined structures using PyRosetta
Step 5: Simulate protein-protein interactions using HADDOCK
Step 6: Visualize the final structures using PyMOL
