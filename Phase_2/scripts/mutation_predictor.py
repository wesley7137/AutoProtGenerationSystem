import random

class SimpleMutationPredictor:
    def __init__(self):
        self.hydrophobic = set('AVILMFYW')
        self.polar = set('STNQ')
        self.charged = set('RHKDE')
        self.special = set('CGP')

    def predict_mutation_effects(self, sequence, mutations):
        mutation_scores = {}
        for mutation in mutations:
            wt_aa, position, mt_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
            score = self._score_mutation(wt_aa, mt_aa)
            mutation_scores[mutation] = score
        return mutation_scores

    def _score_mutation(self, wt_aa, mt_aa):
        if wt_aa == mt_aa:
            return 0
        elif (wt_aa in self.hydrophobic and mt_aa in self.hydrophobic) or \
             (wt_aa in self.polar and mt_aa in self.polar) or \
             (wt_aa in self.charged and mt_aa in self.charged):
            return random.uniform(-0.1, 0.1)
        elif wt_aa in self.special or mt_aa in self.special:
            return random.uniform(-0.5, 0.5)
        else:
            return random.uniform(-0.3, 0.3)