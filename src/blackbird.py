from FixedMCTS import FixedMCTS
import numpy as np

class BlackBird(FixedMCTS):
    """Class to train a network using an MCTS driver to improve decision making"""
    class TrainingExample(object):
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
        # Need to pass in these things to super.
        return super().__init__(maxDepth, explorationRate, 1, timeLimit, playLimit, **kwargs)


    def GenerateTrainingSamples(self, nGames):
        assert nGames > 0, 'What are you doing?'

        examples = set()

        for i in len(nGames):
            gameHistory = set()
            state = self.NewGame()
            lastAction = None
            winner = None
            self.ResetRoot()
            while winner is not None:
                (nextState, currentValue, currentProbabilties) = self.FindMove(state)
                example = TrainingExample(state, None, currentProbabilties)
                state = nextState
                self.MoveRoot(state)

                winner = self.Winner(state, lastAction)
                gameHistory.add(example)
                
            example = TrainingExample(state, None, np.zeros([len(currentProbabilties)]))
            gameHistory.add(example)
            
            for example in gameHistory:
                example.Value = 1 if example.State.Player == winner else 0

            examples |= gameHistory

        return examples


    def LearnFromExamples(self, examples):
        raise NotImplementedError

    # Overriden from MCTS
    def SampleValue(self, state, player):
        raise NotImplementedError

    def GetPriors(self, state):
        raise NotImplementedError

    # Need to be overriden
    def NewGame(self):
        raise NotImplementedError

if __name__ == '__main__':
    b = BlackBird(1,2)