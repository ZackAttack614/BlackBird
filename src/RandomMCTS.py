from MCTS import MCTS
import numpy as np

class RandomMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        return

    def FindMove(self, state, *args, **kwargs):

        action = np.random.choice([i for i in range(len(state.LegalActions())) 
                                    if state.LegalActions()[i] == 1])
        winRate = np.random.random()
        childProbability = state.LegalActions()
        s = sum(childProbability)
        if s > 0:
            childProbability /= s

        return self._applyAction(state, action), winRate, childProbability


    def ResetRoot(self, *args, **kwargs):
        return

    def MoveRoot(self, *args, **kwargs):
        return
