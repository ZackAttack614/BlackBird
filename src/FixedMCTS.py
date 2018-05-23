from MCTS import MCTS
from MCTS import Node
import numpy as np

class FixedMCTS(MCTS):
    """An implementation of Monte Carlo Tree Search that only aggregates statistics up to a fixed depth."""
    def __init__(self, maxDepth, explorationRate, threads = 1, timeLimit = None, playLimit = None, **kwargs):
        self.MaxDepth = maxDepth
        return super().__init__(explorationRate, timeLimit, playLimit, threads, **kwargs)
    
    # Overriding from MCTS
    def RunSimulation(self, root):
        node = root
        lastAction = None
        previousPlayer = None
        for i in range(self.MaxDepth):
            if node.Children is None:
                if self.Winner(node.State, lastAction) is not None:
                    break
                self.AddChildren(node)
            if np.sum(node.LegalActions) == 0:
                break
            previousPlayer = node.State.Player
            lastAction = self.SelectAction(node)
            node = node.Children[lastAction]

        assert i > 0, 'When requesting a move from the MCTS, there is at least one legal option.'

        val, player = self.SampleValue(node.State, previousPlayer)
        self.BackProp(node, val, player)
        return
    
    def SampleValue(self, state, player):
        """Samples the value of the state for the specified player."""
        rolloutState = state
        winner = self.Winner(rolloutState)
        while winner is None:
            actions = np.where(self.LegalActions(rolloutState) == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self.ApplyAction(rolloutState, action)
            winner = self.Winner(rolloutState, action)
        return 0.5 if winner == 0 else 1, winner

    def GetPriors(self, state):
        return np.array([1] * len(self.LegalActions(state)))


