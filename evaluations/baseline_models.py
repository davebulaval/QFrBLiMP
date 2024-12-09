from statistics import mode

import numpy as np


class RandomClassBaselineModel:
    def __init__(self, seed: int = 42):
        self.random_generator = np.random.RandomState(seed=seed)

    def __call__(self, x, labels):
        return self.random_generator.random_integers(0, 1, size=1)[0]


class MajorityVoteModel:
    def __call__(self, votes, labels):
        # We use the mode as the majority vote
        return mode(votes)
