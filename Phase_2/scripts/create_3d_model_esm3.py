import pymol
from pymol import cmd

def create_3d_model_esm3(protein_pdb, prompt_string):
    # Start PyMOL in headless mode with quiet mode enabled
    pymol.finish_launching(['pymol', '-cq'])  # '-cq' for command line only and quiet mode
    
    try:
        # Load the PDB file into PyMOL
        cmd.load(protein_pdb)
        
        # Apply color and visualization settings
        cmd.spectrum("count", "rainbow", "all")
        cmd.show("cartoon")
        cmd.bg_color("black")
        
        # Save the image
        image_filename = f"{prompt_string}_model.png"
        cmd.png(image_filename)
        print(f"INFO: Image saved as {image_filename}")
        
        # Save the PyMOL session if needed
        session_filename = f"{prompt_string}_model.pse"
        cmd.save(session_filename)
        print(f"INFO: PyMOL session saved as {session_filename}")
        
    except pymol.CmdException as e:
        print(f"ERROR: PyMOL command failed - {e}")
    finally:
        cmd.quit()

