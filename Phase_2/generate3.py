import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import logging
import os
from datetime import datetime
import csv
import json
import os.path as osp   

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large",
                                             torch_dtype=torch.float16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-large", trust_remote_code=True)
logger.info("Model and tokenizer loaded")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set pad_token to eos_token")

# Define valid amino acids
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def read_technical_descriptions(base_dir):
    file_path = osp.join(base_dir, "technical_descriptions.json")
    if osp.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No technical descriptions found at {file_path}")

def extract_amino_acids(sequence):
    """Extract only valid amino acids from the sequence."""
    return ''.join(char for char in sequence if char in VALID_AMINO_ACIDS)

async def generate_protein_sequence(input_text):
    """Generate a protein sequence based on the input description."""
    model.eval()
    max_length = 256
    input_text = f"Input Text: {input_text} Sequence:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    logger.info(f"Generating sequence for input: {input_text}")

    try:
        with torch.no_grad():
            output = await asyncio.to_thread(
                model.generate,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
        sequence = generated_sequence.split("Sequence:")[-1].strip()
        extracted_sequence = extract_amino_acids(sequence)
        logger.info(f"Generated and extracted sequence: {extracted_sequence[:50]}...")  # Log first 50 amino acids
        print("Extracted Sequence:", extracted_sequence)
        return extracted_sequence
    except Exception as e:
        logger.error(f"Error in generate_protein_sequence: {str(e)}")
        return None

async def save_results_to_csv(input_text, sequences, scores, start_time, output_dir="results"):
    """Save generated sequences to a CSV file with a dynamic filename."""
    os.makedirs(output_dir, exist_ok=True)

    avg_score = sum(scores) / len(scores) if scores else 0
    grade = 'A' if avg_score >= 0.8 else 'B' if avg_score >= 0.6 else 'C' if avg_score >= 0.4 else 'D'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"protein_discovery_{grade}_{avg_score:.2f}_{timestamp}.csv"
    full_path = os.path.join(output_dir, filename)

    def write_csv():
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Score', 'Prompt', 'Time', 'Sequence', 'Keywords'])

            for sequence, score in zip(sequences, scores):
                writer.writerow([
                    score,
                    input_text,
                    start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    sequence,
                    ''  # Keywords left empty as per your request
                ])

    await asyncio.to_thread(write_csv)
    logger.info(f"Results saved to {full_path}")
    return full_path

"""async def main():
    input_text = "A peptide that increases growth activity in telomeres"
    start_time = datetime.now()
    
    technical_descriptions = read_technical_descriptions("Phase_1/molecule_generation")
    sequence = await generate_protein_sequence(technical_descriptions)
    
    # Check if a sequence was generated
    if sequence:
        # Note: We're not using placeholder scores anymore. You should replace this with actual scores from prediction steps.
        scores = []  # This should be populated with actual scores
        await save_results_to_csv(input_text, [sequence], scores, start_time)
        logger.info(f"Generated sequence: {sequence}")
        return sequence
    else:
        logger.error("No valid sequence was generated.")

if __name__ == "__main__":
    asyncio.run(main())"""