import torch
import logging

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ESM3MutationGuide:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info("Loading ESM3 model...")
        try:
            import esm
            self.model = esm.pretrained.esm2_t33_650M_UR50D()  # Use a different ESM model
            self.model = self.model.to(device)
        except Exception as e:
            logger.error(f"Failed to load ESM3 model: {str(e)}")
            self.model = None

    def unload_model(self):
        logger.info("Unloading ESM3 model...")
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()