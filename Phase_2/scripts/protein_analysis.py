from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import truncnorm
import logging

logger = logging.getLogger(__name__)

def predict_protein_function(sequence):
    try:
        sequence = ''.join(sequence.split())
        if not sequence:
            logger.error("Empty sequence for protein function prediction")
            return 0.5

        analysis = ProteinAnalysis(sequence)
        molecular_weight = analysis.molecular_weight()
        aromaticity = analysis.aromaticity()
        instability_index = analysis.instability_index()
        isoelectric_point = analysis.isoelectric_point()
        
        norm_weight = truncnorm.cdf((molecular_weight - 25000) / 10000, -2, 2)
        norm_aromaticity = aromaticity
        norm_instability = 1 - truncnorm.cdf((instability_index - 40) / 10, -2, 2)
        norm_isoelectric = truncnorm.cdf((isoelectric_point - 7) / 2, -2, 2)

        aa_count = {aa: sequence.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        total_aa = len(sequence)
        composition_balance = 1 - sum(abs(count/total_aa - 0.05) for count in aa_count.values()) / 2

        weights = [0.25, 0.15, 0.25, 0.15, 0.2]
        score = sum(w * v for w, v in zip(weights, [norm_weight, norm_aromaticity, norm_instability, norm_isoelectric, composition_balance]))
        return max(0, min(1, score))
    except Exception as e:
        logger.error(f"Error in predict_protein_function: {str(e)}")
        return 0.5