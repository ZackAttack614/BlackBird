import numpy as np
from time import time
import multiprocessing as mp
from GameState import GameState

class Node(object):
    """ This is the abtract tree node class that is used to cache/organize
        game information during the search.
    """
    def __init__(self, state, legalActions, priors, **kwargs):
        self.State = state
        self.Value = 0
        self.Plays = 0
        self.LegalActions = np.array(legalActions)
        self.Children = None
        self.Parent = None

        # Use the legal actions mask to ignore priors that don't make sense.
        self.Priors = np.multiply(priors, legalActions)

        # Do some caching here. This is to reduce the strain on the CPU memory
        # cache compared to receating a new array on every access.
        self._childWinRates = np.zeros(len(legalActions))
        self._childPlays = np.zeros(len(legalActions))

    def WinRate(self):
        return self.Value/self.Plays if self.Plays > 0 else 0

    def ChildProbability(self):
        allPlays = sum(self.ChildPlays())
        zeroProbs = np.zeros((len(self.ChildPlays())))
        return self.ChildPlays() / allPlays if allPlays > 0 else zeroProbs

    def ChildWinRates(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childWinRates[i] = self.Children[i].WinRate()
        return self._childWinRates

    def ChildPlays(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childPlays[i] = self.Children[i].Plays
        return self._childPlays

class MCTS(object):
    """ Base class for Monte Carlo Tree Search algorithms.

        Outlines all the necessary operations for the core MCTS algorithm.
        FindLeaf() will need to be overriden to avoid a NotImplemenetedError.

        Attributes:
            TimeLimit: The default max move time in seconds.
            PlayLimit: The default number of positions to evaluate per move.
            ExplorationRate: The exploration parameter for MCTS.
            Root: The Node object representing the root of the MCTS.
    """
    def __init__(self, explorationRate, timeLimit=None, playLimit=None,
            **kwargs):

        self.TimeLimit = timeLimit
        self.PlayLimit = playLimit
        self.ExplorationRate = explorationRate
        self.Root = None

    def FindMove(self, state, temp=0.1, moveTime=None, playLimit=None):
        """ Finds the optimal move in a position.

            Given a game state, this will use a Monte Carlo Tree Search
            algorithm to pick the best next move.

            Args:
                state: A GameState object which the function will evaluate.
                temp: A float determining the temperature to apply in move
                    selection.
                moveTime: An optional float determining the allowed search time.
                playLimit: An optional float determining the allowed number of
                    positions to evaluate.

            Returns:
                A tuple providing, in order...
                    - The board state after applying the selected move
                    - The decided value of input state
                    - The probabilities of choosing each of the children

            Raises:
                TypeError: state was not an object of type GameState.
                ValueError: The function was not able to determine a stop time.
        """

        if not isinstance(state, GameState):
            raise TypeError('State not of type GameState')

        endTime = None
        if moveTime is None:
            moveTime = self.TimeLimit
        if moveTime is not None:
            endTime = time() + moveTime
        if playLimit is None:
            playLimit = self.PlayLimit
        
        if endTime is None and playLimit is None:
            raise ValueError('Not enough information to decide a stop time.')

        if self.Root is None:
            self.Root = Node(state, state.LegalActions(), self.GetPriors(state))
        assert self.Root.State == state, 'Primed for the correct input state.'

        self._runMCTS(temp, endTime, playLimit)
        action = self._selectAction(self.Root, temp, exploring=False)

        return (self._applyAction(state, action), self.Root.WinRate(),
            self.Root.ChildProbability())

    def _runMCTS(self, temp, endTime=None, nPlays=None):
        """ Run the MCTS algorithm on the current Root Node.

            Given the current game state, represented by self.Root, a child node
            is seleted using the FindLeaf method. This method will apply temp to
            all child node move selection proportions, compute the sampled value
            of the action, and backpropogate the value through the tree.

            Args:
                temp: A float determining the temperature to apply in FindMove.
                endTime: (optional) The maximum time to spend on searching.
                nPlays: (optional) The maximum number of positions to evaluate.
        """

        endPlays = self.Root.Plays + (nPlays if nPlays is not None else 0)
        while ((endTime is None or (time() < endTime or self.Root.Children is None))
                and (nPlays is None or self.Root.Plays < endPlays)):
            node = self.FindLeaf(self.Root, temp)

            val = self.SampleValue(node.State, node.State.PreviousPlayer)
            self._backProp(node, val, node.State.PreviousPlayer)

    def _mergeAll(self, target, trees):
        for t in trees:
            target.Plays += t.Plays
            target.Value += t.Value

        continuedTrees = [t for t in trees if t.Children is not None]
        if len(continuedTrees) == 0:
            return
        if target.Children is None:
            t = continuedTrees[0]
            target.Children = t.Children
            t.Children = None
            for c in target.Children:
                if c is not None:
                    c.Parent = target
            del continuedTrees[0]

        for i in range(len(target.Children)):
            if target.Children[i] is None:
                continue
            self._mergeAll(
                target.Children[i], [t.Children[i] for t in continuedTrees])

    def _selectAction(self, root, temp, exploring=True):
        """ Chooses an action from an explored root.

            Selects a child of the root using an upper confidence interval. If
            you are not exploring, setting the exploring flag to false will
            instead choose the one with the highest expected payout - ignoring 
            the exploration/regret factor.

            Args:
                root: A Node object which must have children Nodes.
                temp: The temperature to apply to the children Node visit
                    counts. If temp is 0, _selectAction will return the child
                    Node with the greatest visit count.
                exploring: A boolean toggle for overriding the selection type to
                    a simple argmax.  If True, _selectAction will return the
                    child Node with the greatest visit count.

            Returns:
                choice: An int representing the index of the selected action.
        """
        assert root.Children is not None, 'The node has children to select.'

        if exploring or temp == 0:
            allPlays = sum(root.ChildPlays())
            upperConfidence = (root.ChildWinRates()
                + (self.ExplorationRate * root.Priors * np.sqrt(1.0 + allPlays))
                / (1.0 + root.ChildPlays()))
            choice = np.argmax(upperConfidence)
            p = None
        else:
            allPlays = sum([p ** (1 / temp) for p in root.ChildPlays()])
            p = [c ** (1 / temp) / allPlays for c in root.ChildPlays()]
            choice = np.random.choice(len(root.ChildPlays()), p=p)

        assert root.LegalActions[choice] == 1, 'Selected move is legal.'
        return choice

    def AddChildren(self, node):
        """ Expands a node and adds children, actions and priors.

            Given a node, MCTS will evaluate the node's children, if they exist.
            The evaluation and prior policy are supplied in the creation of the
            child Node object.

            Args:
                node: A Node object to expand.
        """
        numLegalMoves = len(node.LegalActions)
        node.Children = [None] * numLegalMoves
        for actionIndex in range(numLegalMoves):
            if node.LegalActions[actionIndex] == 1:
                s = self._applyAction(node.State, actionIndex)
                node.Children[actionIndex] = Node(s, s.LegalActions(),
                    self.GetPriors(s))
                node.Children[actionIndex].Parent = node

    def MoveRoot(self, states):
        """ Updates the root of the tree.

            Move the root of the tree to the provided state. Use this to update
            the root so that tree integrity can be maintained between moves if
            necessary. Does nothing if Root is None, for example after running
            DropRoot().

            Args:
                states: A list of Node objects to cycle through, updating the
                    Root as it is iterated over.
        """
        for s in states: 
            self._moveRoot(s)

    def _moveRoot(self, state):
        if self.Root is None:
            return
        if self.Root.Children is None:
            self.Root = None
            return
        for child in self.Root.Children:
            if child is None:
                continue
            if child.State == state:
                self.Root = child
                break

    def ResetRoot(self):
        if self.Root is None:
            return
        while self.Root.Parent is not None:
            self.Root = self.Root.Parent

    def DropRoot(self):
        self.Root = None

    def _backProp(self, leaf, stateValue, playerForValue):
        leaf.Plays += 1
        if leaf.Parent is not None:
            if leaf.Parent.State.Player == playerForValue:
                leaf.Value += stateValue
            else:
                leaf.Value += 1 - stateValue

            self._backProp(leaf.Parent, stateValue, playerForValue)

    def _applyAction(self, state, action):
        s = state.Copy()
        s.ApplyAction(action)
        return s

    '''Can override these'''
    '''Algorithm implementation functions'''
    def GetPriors(self, state):
        """ Gets the array of prior search probabilities. 
            Default is just 1 for each possible move.
        """
        return np.array([1] * len(state.LegalActions()))

    def SampleValue(self, state, player):
        """Samples the value of the state for the specified player.
            Must return the value in [0, 1]
            Default is to randomly playout the game.
        """
        rolloutState = state
        winner = rolloutState.Winner()
        while winner is None:
            actions = np.where(rolloutState.LegalActions() == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self._applyAction(rolloutState, action)
            winner = rolloutState.Winner(action)
        return 0.5 if winner == 0 else int(player == winner)

    '''Must override these'''
    def FindLeaf(self, node):
        raise NotImplementedError

    '''Overriden from Object'''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict

if __name__=='__main__':
    mcts = MCTS(1, np.sqrt(2))
    print(mcts.TimeLimit)
