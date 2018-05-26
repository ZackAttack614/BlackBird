from mcts import MCTS,Node
import numpy as np

class FixedMCTS(MCTS):
    """ An implementation of Monte Carlo Tree Search that only aggregates 
        statistics up to a fixed depth.
    """
    def __init__(self, parameters):
        self.parameters = parameters['mcts']
        self.MaxDepth = self.parameters['maxDepth']
        explorationRate = self.parameters['explorationRate']
        timeLimit = self.parameters['timeLimit']
        playLimit = self.parameters['playLimit']
        threads = self.parameters['threads']

        assert self.MaxDepth > 0, 'MaxDepth for MCTS must be > 0.'

        super().__init__(explorationRate, timeLimit, playLimit, threads)
    
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


