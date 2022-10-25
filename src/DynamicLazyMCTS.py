from LazyMCTS import LazyMCTS, LazyNode
import numpy as np

class DynamicLazyMCTS(LazyMCTS):
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

        return node








            # if node.LegalActions == 0:
            #     break
            # if node.Children is None:
            #     if node.State.Winner(lastAction) is not None:
            #         break
            #     firstMove = np.argmax(node.State.LegalActions())
            #     self.AddChild(node, firstMove)
            # lastAction = self._selectAction(node, temp)
            # if lastAction == -1:
            #     choices = node.State.LegalActions()
            #     for child in node.Children:
            #         choices[child.id] -= 1.0
            #     firstMove = np.argmax(choices)
            #     self.AddChild(node, firstMove)
            # node = node.Children[lastAction]
