import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to generate protein sequences

model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-large", trust_remote_code=True)

# Set the device
model.to(device)

def generate_protein_sequence(input_text, max_length=512):
    model.eval()
    input_text = f"Input Text: {input_text} Sequence:"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
   
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=5, do_sample=True, top_p=0.95)
   
    generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated sequence: {generated_sequence}")
    return generated_sequence.split("Sequence:")[-1].strip()


# Example usage
new_description = "A peptide that increases growth activity in telomeres"
generated_sequence = generate_protein_sequence(new_description)
print(f"Generated sequence for '{new_description}':")
print(generated_sequence)