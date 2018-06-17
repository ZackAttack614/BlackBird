from DynamicMCTS import DynamicMCTS as MCTS
from RandomMCTS import RandomMCTS
from TicTacToe import BoardState
from FixedMCTS import FixedMCTS
from Network import Network

import functools
import random
import yaml
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)

class BlackBird(MCTS, Network):
    """ Class to train a network using an MCTS driver to improve decision making
    """
    class TrainingExample(object):
        def __init__(self, state, value, childValues, probabilities, priors):
            self.State = state
            self.Value = value
            self.ChildValues = childValues.reshape((3,3)) if childValues is not None else None
            self.Reward = None
            self.Priors = priors.reshape((3,3))
            self.Probabilities = probabilities
            return

        def __str__(self):
            state = str(self.State)
            value = 'Value: {}'.format(self.Value)
            childValues = 'Child Values: \n{}'.format(self.ChildValues)
            reward = 'Reward:\n{}'.format(self.Reward)
            probs = 'Probabilities:\n{}'.format(
                self.Probabilities.reshape((3,3)))
            priors = '\nPriors:\n{}\n'.format(self.Priors)
            
            return '\n'.join([state, value, childValues, reward, probs, priors])

    def __init__(self, tfLog=False, loadOld=False, teacher=False, **parameters):
        self.bbParameters = parameters
        self.batchSize = parameters.get('network').get('training').get('batch_size')
        self.learningRate = parameters.get('network').get('training').get('learning_rate')
        
        mctsParams = parameters.get('mcts')
        MCTS.__init__(self, **mctsParams)
        
        networkParams = parameters.get('network')
        Network.__init__(self, tfLog, loadOld=loadOld, teacher=teacher,
            **networkParams)

    def GenerateTrainingSamples(self, nGames, temp):
        assert nGames > 0, 'Use a positive integer for number of games.'

        examples = []

        for _ in range(nGames):
            gameHistory = []
            state = BoardState()
            lastAction = None
            winner = None
            self.DropRoot()
            while winner is None:
                (nextState, v, currentProbabilties) = self.FindMove(state, temp)
                childValues = self.Root.ChildWinRates()
                example = self.TrainingExample(state, 1 - v, childValues,
                    currentProbabilties, priors = self.Root.Priors)
                state = nextState
                self.MoveRoot([state])

                winner = state.Winner(lastAction)
                gameHistory.append(example)
                
            example = self.TrainingExample(state, None, None,
                np.zeros([len(currentProbabilties)]),
                np.zeros([len(currentProbabilties)]))
            gameHistory.append(example)
            
            for example in gameHistory:
                if winner == 0:
                    example.Reward = 0
                else:
                    example.Reward = 1 if example.State.Player == winner else -1

            examples += gameHistory

        return examples

    def LearnFromExamples(self, examples, teacher=None):
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
                self.learningRate,
                teacher
                )
        return

    def TestRandom(self, temp, numTests):
        return self.Test(RandomMCTS(), temp, numTests)

    def TestPrevious(self, temp, numTests):
        oldBlackbird = BlackBird(tfLog=False, loadOld=True,
            **self.bbParameters)

        wins, draws, losses = self.Test(oldBlackbird, temp, numTests)

        del oldBlackbird
        return wins, draws, losses

    def TestGood(self, temp, numTests):
        good = FixedMCTS(maxDepth = 10, explorationRate = 0.85, timeLimit = 1)
        return self.Test(good, temp, numTests)

    def Test(self, other, temp, numTests):
        wins = draws = losses = 0

        for _ in range(numTests):
            blackbirdToMove = random.choice([True, False])
            blackbirdPlayer = 1 if blackbirdToMove else 2
            winner = None
            self.DropRoot()
            other.DropRoot()
            state = BoardState()
            
            while winner is None:
                if blackbirdToMove:
                    (nextState, *_) = self.FindMove(state, temp)
                else:
                    (nextState, *_) = other.FindMove(state, temp)
                state = nextState
                self.MoveRoot([state])
                other.MoveRoot([state])

                blackbirdToMove = not blackbirdToMove
                winner = state.Winner()

            if winner == blackbirdPlayer:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

        return wins, draws, losses

    # Overriden from MCTS
    @functools.lru_cache(maxsize = 4096)
    def SampleValue(self, state, player):
        value = self.getEvaluation(state.AsInputArray())
        value = (value + 1 ) * 0.5 # [-1, 1] -> [0, 1]
        if state.Player != player:
            value = 1 - value
        assert value >= 0, 'Value: {}'.format(value)
        return value

    @functools.lru_cache(maxsize = 4096)
    def GetPriors(self, state):
        policy = self.getPolicy(state.AsInputArray()) * state.LegalActions()
        policy /= np.sum(policy)
        
        return policy

if __name__ == '__main__':
    with open('parameters.yaml', 'r') as param_file:
        parameters = yaml.load(param_file)
    b = BlackBird(tfLog=True, loadOld=True, **parameters)

    for i in range(parameters.get('selfplay').get('epochs')):
        examples = b.GenerateTrainingSamples(
            parameters.get('selfplay').get('training_games'),
            parameters.get('mcts').get('temperature').get('exploration'))
        for e in examples:
            print(e)
        b.LearnFromExamples(examples)
