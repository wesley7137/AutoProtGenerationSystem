import torch
from transformers import BertTokenizer, BertForMaskedLM
import logging

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProtBERTAnalyzer:
    def __init__(self):
        self.model_name = "EvaKlimentova/knots_protbertBFD_alphafold"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        logger.info("Loading ProtBERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(self.model_name)
        self.model = self.model.to(device)
        self.model.eval()

    def analyze_sequence(self, protein_sequence):
        if self.model is None:
            self.load_model()
        protein_sequence = ''.join(protein_sequence.split())
        inputs = self.tokenizer(protein_sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def convert_ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return [token for token in tokens if token in 'ACDEFGHIKLMNPQRSTVWY']

    def unload_model(self):
        logger.info("Unloading ProtBERT model...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()