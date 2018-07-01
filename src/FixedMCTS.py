from MCTS import MCTS, Node
import numpy as np

class FixedMCTS(MCTS):
    """ An implementation of Monte Carlo Tree Search that only aggregates 
        statistics up to a fixed depth.
    """
    def __init__(self, **kwargs):
        self.MaxDepth = kwargs.get('maxDepth')
        explorationRate = kwargs.get('explorationRate')
        timeLimit = kwargs.get('timeLimit')
        playLimit = kwargs.get('playLimit')
        threads = kwargs.get('threads', 1) 

        if self.MaxDepth <= 0:
            raise ValueError('MaxDepth for MCTS must be > 0.')

        super().__init__(explorationRate, timeLimit, playLimit)

    # Overriding from MCTS
    def _findLeaf(self, node, temp):
        lastAction = None
        for _ in range(self.MaxDepth):
            if node.Children is None:
                if node.State.Winner(lastAction) is not None:
                    break
                self.AddChildren(node)
            if np.sum(node.LegalActions) == 0:
                break
            lastAction = self._selectAction(node, temp)
            node = node.Children[lastAction]
        assert lastAction is not None, 'There is at least one legal option.'

        return node
