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
        self.batchSize = parameters.get('network').get('training').get('batch_size')
        self.learningRate = parameters.get('selfplay').get('learning_rate')
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
                (nextState, _, currentProbabilties) = self.FindMove(state)
                example = self.TrainingExample(state, None, currentProbabilties)
                state = nextState
                self.MoveRoot([state])

                winner = state.Winner(lastAction)
                gameHistory.append(example)
                
            example = self.TrainingExample(state, None, np.zeros([len(currentProbabilties)]))
            gameHistory.append(example)
            
            for example in gameHistory:
                if winner == 0:
                    example.Value = 0.5 # Draw
                else:
                    example.Value = 1 if example.State.Player == winner else -1

            examples += gameHistory

        return examples

    def LearnFromExamples(self, examples):
        examples = np.random.choice(examples, 
                                    len(examples) - (len(examples) % self.batchSize), 
                                    replace = False)
                            
        for i in range(len(examples) // self.batchSize):
            start = i * self.batchSize
            batch = examples[start : start + self.batchSize]
            self.train(
                    np.stack([b.State.AsInputArray()[0] for b in batch], axis = 0),
                    np.stack([b.Value for b in batch], axis = 0),
                    np.stack([b.Probabilities for b in batch], axis = 0),
                    self.learningRate
                    )
        return

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
    print(b.GenerateTrainingSamples(10))
    raise Exception('Done')

    for i in range(20):
        b.LearnFromExamples(b.GenerateTrainingSamples(10))
    for t in b.GenerateTrainingSamples(1):
        print(t.State)
        print(t.Probabilities)
