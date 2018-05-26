from mcts import MCTS,Node
import numpy as np

class FixedMCTS(MCTS):
    """An implementation of Monte Carlo Tree Search that only aggregates statistics up to a fixed depth."""
    def __init__(self, maxDepth, explorationRate, threads = 1, timeLimit = None, playLimit = None, **kwargs):
        self.MaxDepth = maxDepth
        return super().__init__(explorationRate, timeLimit, playLimit, threads, **kwargs)
    
    # Overriding from MCTS
    def FindLeaf(self, node):
        lastAction = None
        for i in range(self.MaxDepth):
            if node.Children is None:
                if node.State.Winner(lastAction) is not None:
                    break
                self.AddChildren(node)
            if np.sum(node.LegalActions) == 0:
                break
            lastAction = self._selectAction(node)
            node = node.Children[lastAction]
        assert i > 0, 'When requesting a move from the MCTS, there is at least one legal option.'
        return node


