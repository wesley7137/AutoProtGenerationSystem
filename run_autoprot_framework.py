import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import asyncio
from Phase_1.Phase1 import run_Phase_1
from Phase_2.Phase2 import run_Phase_2
from datetime import datetime
import logging
from utils.save_utils import create_organized_directory_structure, create_sequence_directories, save_results_to_csv, save_partial_results


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoProt Framework: Novel protein generation and analysis pipeline")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Input text describing the desired protein function")
    parser.add_argument("--max_generations", type=int, default=2,
                        help="Maximum number of generations for technical descriptions")
    parser.add_argument("--num_reflections", type=int, default=2,
                        help="Number of reflection rounds for technical descriptions")
    parser.add_argument("--num_sequences", type=int, default=2,
                        help="Number of protein sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, default=2,
                        help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, default=0.7,
                        help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--base_dir", type=str, default="molecule_generation",
                        help="Directory containing the technical descriptions")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--skip_description_gen", action="store_true",
                        help="Skip technical description generation and use input text directly")
    
    return parser.parse_args()




async def run_autoprot_framework():
    args = parse_arguments()
    start_time = datetime.now()
    
    logger.info(f"Starting AutoProt Framework at {start_time}")
    logger.info(f"User Prompt: {args.input_text}")

    # Create the main run directory
    run_dir, predicted_structures_dir, results_dir = create_organized_directory_structure(args.output_dir)

    # Phase 1: Generate technical descriptions (if not skipped)
    if not args.skip_description_gen:
        logger.info("\nStarting Phase 1: Generating technical descriptions")
        technical_descriptions = run_Phase_1(args.input_text, args.max_generations, args.num_reflections)
        logger.info(f"Generated {len(technical_descriptions)} technical descriptions")
    else:
        logger.info("\nSkipping Phase 1: Using input text as technical description")
        technical_descriptions = [{"technical_instruction": args.input_text}]

    # Phase 2: Generate and analyze novel proteins
    logger.info("\nStarting Phase 2: Generating and analyzing novel proteins")
    pipeline_results = await run_Phase_2(
        args.input_text,
        args.optimization_steps,
        args.score_threshold,
        run_dir,
        results_dir,
        predicted_structures_dir,
        technical_descriptions
    )

    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"\nAutoProt Framework completed at {end_time}")
    logger.info(f"Total execution time: {total_time}")

    return pipeline_results


if __name__ == "__main__":
    asyncio.run(run_autoprot_framework())