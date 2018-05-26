from FixedMCTS import FixedMCTS
from TicTacToe import BoardState
from network import Network

import yaml
import numpy as np

class BlackBird(FixedMCTS, Network):
    """ Class to train a network using an MCTS driver to improve decision making
    """
    class TrainingExample(object):
        def __init__(self, state, value, probabilities):
            self.State = state # state holds the player
            self.Value = value
            self.Probabilities = probabilities
            return

    def __init__(self, parameters):
        FixedMCTS.__init__(self, parameters=parameters)
        Network.__init__(self, parameters=parameters)

    def GenerateTrainingSamples(self, nGames):
        assert nGames > 0, 'Use a positive integer for number of games.'

        examples = []

        for i in range(nGames):
            gameHistory = []
            state = BoardState()
            lastAction = None
            winner = None
            self.ResetRoot()
            while winner is None:
                (nextState, currentValue, currentProbabilties) = self.FindMove(state)
                example = self.TrainingExample(state, None, currentProbabilties)
                state = nextState
                self.MoveRoot([state])

                winner = state.Winner(lastAction)
                gameHistory.append(example)
                
            example = self.TrainingExample(state, None, np.zeros([len(currentProbabilties)]))
            gameHistory.append(example)
            
            for example in gameHistory:
                example.Value = 1 if example.State.Player == winner else 0

            examples += gameHistory

        return examples

    def LearnFromExamples(self, examples):
        raise NotImplementedError

    # Overriden from MCTS
    def SampleValue(self, state, player):
        value = self.getEvaluation(state.AsInputArray())
        return value, player

    def GetPriors(self, state):
        return self.getPolicy(state.AsInputArray())

if __name__ == '__main__':
    with open('parameters.yaml', 'r') as param_file:
        parameters = yaml.load(param_file)
    b = BlackBird(parameters)
    b.GenerateTrainingSamples(1)
