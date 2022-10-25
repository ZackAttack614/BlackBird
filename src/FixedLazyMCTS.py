from LazyMCTS import LazyMCTS, LazyNode
import numpy as np

class FixedLazyMCTS(LazyMCTS):
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
            if node.LegalActions == 0:
                break
            if node.Children is None:
                if node.State.Winner(lastAction) is not None:
                    break
                fullChildIds = node.PriorIndices
                assert len(fullChildIds) > 0, 'There is at least one child to add.'
                firstMove = fullChildIds[0]
                self.AddChild(node, firstMove)
            lastAction = self._selectAction(node, temp)
            if lastAction not in node._childIds:
                self.AddChild(node, node.PriorIndices[lastAction])
                node = node.Children[-1]
            else:
                node = node.Children[node._childIds.index(lastAction)]
        assert lastAction is not None, 'There is at least one legal option.'

        return node
