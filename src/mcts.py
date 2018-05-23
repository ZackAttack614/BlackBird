import numpy as np
import time
import multiprocessing as mp


def softMax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


class Node:
    def __init__(self, state, legalActions, priors, **kwargs):
        self.State = state
        self.Value = 0
        self.Plays = 0
        self.LegalActions = legalActions
        self.Children = None
        self.Parent = None
        # Use the legal actions mask to ignore priors that don't make sense.
        self.Priors = np.multiply(priors, legalActions)

        # Do some caching here. This is to reduce the strain on the CPU memory cache compared to receating a new array on every access.
        self._childWinRates = np.zeros(len(legalActions))
        self._childPlays = np.zeros(len(legalActions))

        return super().__init__(**kwargs)

    def WinRate(self):
        return self.Value/self.Plays if self.Plays > 0 else 0.

    def ChildWinRates(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childWinRates[i] = self.Children[i].WinRate()
        return self._childWinRates

    def ChildProbability(self):
        rates = np.array(self.ChildWinRates())
        
        return softMax(rates)

    def ChildPlays(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childPlays[i] = self.Children[i].Plays
        return self._childPlays

class MCTS:
    """This is a base class for Monte Carlo Tree Search algorithms. It outlines all the necessary operations for the core algorithm.
        Most operations will need to be overriden to avoid a NotImplemenetedError."""
    def __init__(self, explorationRate, timeLimit = None, playLimit = None, threads = 1, **kwargs):
        self.TimeLimit = timeLimit
        self.PlayLimit = playLimit
        self.ExplorationRate = explorationRate
        self.Root = None
        self.Threads = threads
        if self.Threads > 1:
            self.Pool = mp.Pool(processes = self.Threads)
        return super().__init__(**kwargs)

    def FindMove(self, state, moveTime = None, playLimit = None):
        """Given a game state, this will use a Monte Carlo Tree Search algorithm to pick the best next move."""
        endTime = None
        if moveTime is None:
            moveTime = self.TimeLimit
        if moveTime is not None:
            endTime = time.time() + moveTime
        if playLimit is None:
            playLimit = self.PlayLimit

        if self.Root is None:
            self.Root = Node(state, self.LegalActions(state), self.GetPriors(state))

        assert self.Root.State == state, 'MCTS has been primed for the correct input state.'
        assert endTime is not None or playLimit is not None, 'The MCTS algorithm has a cutoff point.'
        
        if self.Threads == 1:
            self._runMCTS(self.Root, endTime, playLimit)
        elif self.Threads > 1:
            self._runAsynch(state, endTime, playLimit)

        return self.ApplyAction(state, self.SelectAction(self.Root, True))

    def _runAsynch(self, state, endTime = None, nPlays = None):
        roots = []
        results = []
        for i in range(self.Threads):
            root = Node(state, self.LegalActions(state), self.GetPriors(state))
            results.append(self.Pool.apply_async(self._runMCTS, (root, endTime, nPlays)))

        for r in results:
            roots.append(r.get())

        self._mergeAll(self.Root, roots)
        return

    def _runMCTS(self, root, endTime = None, nPlays = None):
        while (endTime is None or time.time() < endTime) and (nPlays is None or root.Plays < nPlays):
            self.RunSimulation(root)
        for c in root.Children:
            if c is not None:
                c.Children = None # Kill the children. We only want to pass back the immediate results.
        return root

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
            self._mergeAll(target.Children[i], [t.Children[i] for t in continuedTrees])

        return

    def SelectAction(self, root, testing = False):
        """Selects a child of the root using an upper confidence interval. If you are not exploring, setting the testing flag will
            instead choose the one with the highest expected payout - ignoring the exploration/regret factor."""
        assert root.Children is not None, 'The node has children to select.'

        upperConfidence = root.ChildWinRates()
        if not testing:
            upperConfidence += self.ExplorationRate * root.Priors * np.sqrt(root.Plays) / (1.0 + root.ChildPlays())

        return np.argmax(upperConfidence + root.LegalActions)

    def AddChildren(self, node):
        """Expands the node and adds children, actions and priors."""
        l = len(node.LegalActions)
        node.Children = [None] * l
        for i in range(l):
            if node.LegalActions[i] == 1:
                s = self.ApplyAction(node.State, i)
                node.Children[i] = Node(s, self.LegalActions(s), self.GetPriors(s))
                node.Children[i].Parent = node
        return

    def MoveRoot(self, states):
        for s in states: 
            self._moveRoot(s)
        return

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
        return

    def ResetRoot(self):
        if self.Root is None:
            return
        while self.Root.Parent is not None:
            self.Root = self.Root.Parent
        return

    def DropRoot(self):
        self.Root = None
        return

    def BackProp(self, leaf, stateValue, playerForValue):
        leaf.Plays += 1
        if leaf.Parent is not None:
            if leaf.Parent.State.Player == playerForValue:
                leaf.Value += stateValue
            else:
                leaf.Value += 1 - stateValue

            self.BackProp(leaf.Parent, stateValue, playerForValue)
        return
    
    '''Algorithm implementation functions'''
    def RunSimulation(self, root):
        raise NotImplementedError
    
    def SampleValue(self, state, player):
        raise NotImplementedError

    def GetPriors(self, state):
        raise NotImplementedError

    '''Game implementation functions.'''
    def ApplyAction(self, state, action):
        raise NotImplementedError

    def LegalActions(self, state):
        raise NotImplementedError

    def Winner(self, state, lastAction = None):
        raise NotImplementedError

    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict

if __name__=='__main__':
    mcts = MCTS(1, np.sqrt(2))
    print(mcts.TimeLimit)