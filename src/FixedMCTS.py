from mcts import MCTS,Node
import numpy as np

class FixedMCTS(MCTS):
    """ An implementation of Monte Carlo Tree Search that only aggregates 
        statistics up to a fixed depth.
    """
    def __init__(self, **kwargs):
        self.parameters = kwargs.get('mcts')
        self.MaxDepth = self.parameters.get('maxDepth')
        explorationRate = self.parameters.get('explorationRate')
        timeLimit = self.parameters.get('timeLimit')
        playLimit = self.parameters.get('playLimit')
        threads = self.parameters.get('threads')
        if threads is None: threads =1 

        assert self.MaxDepth > 0, 'MaxDepth for MCTS must be > 0.'

        super().__init__(explorationRate, timeLimit, playLimit, threads)
    
    # Overriding from MCTS
    def FindLeaf(self, node, temp):
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
