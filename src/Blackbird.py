from DynamicMCTS import DynamicMCTS as MCTS
from RandomMCTS import RandomMCTS
from FixedMCTS import FixedMCTS
from Network import Network

import functools
import random
import yaml
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)

class BlackBird(MCTS, Network):
    """ Class which encapsulates MCTS powered by a neural network.

        The BlackBird class is designed to learn how to win at a board game, by
        using Monte Carlo Tree Search (MCTS) with the tree search powered by a
        neural network.

        Attributes:
            BoardState: A GameState object which holds the rules of the game
                BlackBird is intended to learn.
            bbParameters: A dictionary holding algorithmic parameters, usually
                read in from parameters.yaml in the root of the project.
            boardShape: A list of integers representing the dimension of the
                board game to be learned.
    """
    class TrainingExample(object):
        def __init__(self, state, value, childValues, childValuesStr,
                probabilities, priors, priorsStr, boardShape):

            self.State = state
            self.Value = value
            self.BoardShape = boardShape
            self.ChildValues = childValues if childValues is not None else None
            self.ChildValuesStr = childValuesStr if childValuesStr is not None else None
            self.Reward = None
            self.Priors = priors
            self.PriorsStr = priorsStr
            self.Probabilities = probabilities

        def __str__(self):
            state = str(self.State)
            value = 'Value: {}'.format(self.Value)
            childValues = 'Child Values: \n{}'.format(self.ChildValuesStr)
            reward = 'Reward:\n{}'.format(self.Reward)
            probs = 'Probabilities:\n{}'.format(
                self.Probabilities.reshape(self.BoardShape))
            priors = '\nPriors:\n{}\n'.format(self.PriorsStr)

            return '\n'.join([state, value, childValues, reward, probs, priors])

    def __init__(self, boardState, tfLog=False, loadOld=False, teacher=False,
            **parameters):

        self.BoardState = boardState
        self.bbParameters = parameters
        self.boardShape = boardState().BoardShape

        mctsParams = parameters.get('mcts')
        MCTS.__init__(self, **mctsParams)

        networkParams = parameters.get('network')
        Network.__init__(self, tfLog, self.boardShape, boardState.LegalMoves,
            teacher=teacher, loadOld=loadOld, **networkParams)

    def GenerateTrainingSamples(self, nGames, temp):
        """ Generates self-play games to learn from.

            This method generates `nGames` self-play games, and returns them as
            a list of `TrainingExample` objects.

            Args:
                `nGames`: An int determining the number of games to generate.
                `temp`: A float between 0 and 1 determining the exploration temp
                    for MCTS. Usually this should be close to 1 to ensure
                    high move exploration rate.

            Returns:
                A list of `TrainingExample` objects holding game states in the
                    `nGames` games produced.

            Raises:
                ValueError: nGames was not a positive integer.
        """
        if nGames <= 0:
            raise ValueError('Use a positive integer for number of games.')

        examples = []

        for _ in range(nGames):
            gameHistory = []
            state = self.BoardState()
            lastAction = None
            winner = None
            self.DropRoot()
            while winner is None:
                (nextState, v, currentProbabilties) = self.FindMove(state, temp)
                childValues = self.Root.ChildWinRates()
                example = self.TrainingExample(state, 1 - v, childValues,
                    state.EvalToString(childValues), currentProbabilties, self.Root.Priors, 
                    state.EvalToString(self.Root.Priors), state.LegalActionShape())
                state = nextState
                self.MoveRoot(state)

                winner = state.Winner(lastAction)
                gameHistory.append(example)

            example = self.TrainingExample(state, None, None, None,
                np.zeros([len(currentProbabilties)]),
                np.zeros([len(currentProbabilties)]),
                np.zeros([len(currentProbabilties)]),
                state.LegalActionShape())
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

        batchSize = self.bbParameters.get('network').get('training').get('batch_size')
        examples = np.random.choice(examples, 
            len(examples) - (len(examples) % batchSize), 
            replace = False)

        for i in range(len(examples) // batchSize):
            start = i * batchSize
            batch = examples[start : start + batchSize]
            self.train(
                np.stack([b.State.AsInputArray()[0] for b in batch], axis = 0),
                np.stack([b.Reward for b in batch], axis = 0),
                np.stack([b.Probabilities for b in batch], axis = 0),
                self.bbParameters.get('network').get('training').get('learning_rate'),
                teacher
                )

    def TestRandom(self, temp, numTests):
        return self.Test(RandomMCTS(), temp, numTests)

    def TestPrevious(self, temp, numTests):
        oldBlackbird = BlackBird(self.BoardState, tfLog=False, loadOld=True,
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
            state = self.BoardState()

            while winner is None:
                if blackbirdToMove:
                    (nextState, *_) = self.FindMove(state, temp)
                else:
                    (nextState, *_) = other.FindMove(state, temp)
                state = nextState
                self.MoveRoot(state)
                other.MoveRoot(state)

                blackbirdToMove = not blackbirdToMove
                winner = state.Winner()

            if winner == blackbirdPlayer:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

        return wins, draws, losses

    @functools.lru_cache(maxsize=4096)
    def SampleValue(self, state, player):
        value = self.getEvaluation(state.AsInputArray())
        value = (value + 1 ) * 0.5 # [-1, 1] -> [0, 1]
        if state.Player != player:
            value = 1 - value
        assert value >= 0, 'Value: {}'.format(value)
        return value

    @functools.lru_cache(maxsize=4096)
    def GetPriors(self, state):
        policy = self.getPolicy(state.AsInputArray()) * state.LegalActions()
        policy /= np.sum(policy)

        return policy

if __name__ == '__main__':
    from Connect4 import BoardState
    with open('parameters.yaml', 'r') as param_file:
        parameters = yaml.load(param_file)
    b = BlackBird(BoardState, tfLog=True, loadOld=True, **parameters)

    for i in range(1):
        examples = b.GenerateTrainingSamples(
            1,
            parameters.get('mcts').get('temperature').get('exploration'))
        for e in examples:
            print(e)
