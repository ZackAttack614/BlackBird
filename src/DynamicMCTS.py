from MCTS import MCTS, Node
import numpy as np

class DynamicMCTS(MCTS):
    """ An extension of the MCTS class that aggregates statistics as it
        explores.

        This class is identical to base MCTS, with the class function FindLeaf
        defined.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _findLeaf(self, node, temp):
        """ Applies MCTS to a supplied node until a leaf is found.

            Args:
                node: A Node object to find a leaf of.
                temp: The temperature to apply to action selection after tree
                    search has been applied to the node.
        """
        lastAction = None
        while True:
            if node.Children is None:
                if node.State.Winner(lastAction) is not None:
                    break
                self.AddChildren(node)
                break
            if np.sum(node.LegalActions) == 0:
                break
            lastAction = self._selectAction(node, temp)
            node = node.Children[lastAction]

        return node
