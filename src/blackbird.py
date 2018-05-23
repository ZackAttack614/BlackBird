from FixedMCTS import FixedMCTS
import numpy as np

class BlackBird(FixedMCTS):
    """Class to train a network using an MCTS driver to improve decision making"""
    class TrainingExample():
        def __init__(self, state, value, probabilities):
            self.State = state # state holds the player
            self.Value = value
            self.Probabilities = probabilities
            return

        def asInputArray(self):
            raise NotImplementedError
            return

    def __init__(self, maxDepth, explorationRate, threads = 1, timeLimit = None, playLimit = None, **kwargs):
        raise NotImplementedError
        # Probably want to store the network somewhere in here :).
        # Need to pass in these things.
        return super().__init__(maxDepth, explorationRate, 1, timeLimit, playLimit, **kwargs)


    def GenerateTrainingSamples(self, nGames):
        examples = set()
        maxNMoves = None

        for i in len(nGames):
            gameHistory = set()
            state = self.NewGame()
            lastAction = None
            winner = None
            while winner is not None:
                (nextState, currentValue, currentProbabilties) = self.FindMove(state)
                example = TrainingExample(state, None, currentProbabilties)
                state = nextState
                winner = self.Winner(state, lastAction)
                gameHistory += example
                
                if maxNMoves is None: # Don't want to hard code this, and I don't really want to have a whole property for it. So ill just snipe it from an array.
                    maxNMoves = len(currentProbabilties)
            
            example = TrainingExample(state, None, np.zeros([maxNMoves]))
            gameHistory += example
            
            for example in gameHistory:
                example.Value = 1 if example.sate.Player == winner else 0

            examples |= gameHistory

        return examples


    def LearnExamples(self, examples):
        raise NotImplementedError


    # Overriden from MCTS
    def SampleValue(self, state, player):
        raise NotImplementedError

    def GetPriors(self, state):
        raise NotImplementedError

