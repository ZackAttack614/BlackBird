from DynamicMCTS import DynamicMCTS as MCTS
from TicTacToe import BoardState
from network import Network
import functools

import yaml
import numpy as np
np.set_printoptions(precision=2)

class BlackBird(MCTS, Network):
    """ Class to train a network using an MCTS driver to improve decision making
    """
    class TrainingExample(object):
        def __init__(self, state, value, childValues, probabilities, priors):
            self.State = state # state holds the player
            self.Value = value
            self.ChildValues = childValues.reshape((3,3)) if childValues is not None else None
            self.Reward = None
            self.Priors = priors.reshape((3,3))
            self.Probabilities = probabilities
            return

        def __str__(self):
            return '{}\nValue: {}\nChild Values:\n{}\nReward: {}\nProbabilites:\n{}\n\nPriors:\n{}\n'.format(
                    str(self.State),
                    str(self.Value),
                    str(self.ChildValues),
                    str(self.Reward), 
                    str(self.Probabilities.reshape((3,3))),
                    str(self.Priors)
                    )

    def __init__(self, saver=False, tfLog=False, **parameters):
        self.batchSize = parameters.get('network').get('training').get('batch_size')
        self.learningRate = parameters.get('network').get('training').get('learning_rate')
        MCTS.__init__(self, **parameters)
        Network.__init__(self, saver = saver, tfLog = tfLog, **parameters)

    def GenerateTrainingSamples(self, nGames, temp):
        assert nGames > 0, 'Use a positive integer for number of games.'

        examples = []

        for i in range(nGames):
            gameHistory = []
            state = BoardState()
            lastAction = None
            winner = None
            self.DropRoot()
            while winner is None:
                (nextState, v, currentProbabilties) = self.FindMove(state, temp)
                childValues = self.Root.ChildWinRates()
                example = self.TrainingExample(state, 1 - v, childValues, currentProbabilties, priors = self.Root.Priors)
                state = nextState
                self.MoveRoot([state])

                winner = state.Winner(lastAction)
                gameHistory.append(example)
                
            example = self.TrainingExample(state, None, None, np.zeros([len(currentProbabilties)]), np.zeros([len(currentProbabilties)]))
            gameHistory.append(example)
            
            for example in gameHistory:
                if winner == 0:
                    example.Reward = 0
                else:
                    example.Reward = 1 if example.State.Player == winner else -1

            examples += gameHistory

        return examples

    def LearnFromExamples(self, examples):
        self.SampleValue.cache_clear()
        self.GetPriors.cache_clear()
        
        examples = np.random.choice(examples, 
                                    len(examples) - (len(examples) % self.batchSize), 
                                    replace = False)
                            
        for i in range(len(examples) // self.batchSize):
            start = i * self.batchSize
            batch = examples[start : start + self.batchSize]
            self.train(
                    np.stack([b.State.AsInputArray()[0] for b in batch], axis = 0),
                    np.stack([b.Reward for b in batch], axis = 0),
                    np.stack([b.Probabilities for b in batch], axis = 0),
                    self.learningRate
                    )
        return

    # Overriden from MCTS
    @functools.lru_cache(maxsize = 4096)
    def SampleValue(self, state, player):
        value = self.getEvaluation(state.AsInputArray()) # Gets the value for the current player.
        value = (value + 1 ) * 0.5 # [-1, 1] -> [0, 1]
        if state.Player != player:
            value = 1 - value
        assert value >= 0, 'Value: {}'.format(value) # Just to make sure Im not dumb :).
        return value

    @functools.lru_cache(maxsize = 4096)
    def GetPriors(self, state):
        policy = self.getPolicy(state.AsInputArray()) * state.LegalActions()
        policy /= np.sum(policy)
        
        return policy

if __name__ == '__main__':
    with open('parameters.yaml', 'r') as param_file:
        parameters = yaml.load(param_file)
    b = BlackBird(saver=True, tfLog=True, loadOld=True, **parameters)

    for i in range(parameters.get('selfplay').get('epochs')):
        examples = b.GenerateTrainingSamples(
            parameters.get('selfplay').get('training_games'),
            parameters.get('mcts').get('temperature').get('exploration'))
        for e in examples:
            print(e)
        b.LearnFromExamples(examples)