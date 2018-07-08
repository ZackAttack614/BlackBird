from DynamicMCTS import DynamicMCTS as MCTS
from RandomMCTS import RandomMCTS
from FixedMCTS import FixedMCTS
from Network import Network
from NetworkFactory import NetworkFactory
from DataManager import Connection
from proto.state_pb2 import State

import functools
import random
import yaml
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class ExampleState(object):
    def __init__(self, serialState=None):
        state = State()
        state.ParseFromString(serialState)
        boardDims = np.frombuffer(state.boardDims, dtype=np.int8)

        self.player = state.player,
        self.mctsEval = state.mctsEval,
        self.mctsPolicy = np.frombuffer(state.mctsPolicy,
                                        dtype=np.float),
        self.board = np.frombuffer(state.boardEncoding,
                                   dtype=np.int8).reshape(boardDims)

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


def TestRandom(model, temp, numTests):
    stats = TestModels(model, RandomMCTS(), temp, numTests)
    model.Conn.PutTrainingStatistic('random', stats, model.ModelKey)
    
    return stats


def TestPrevious(model, temp, numTests):
    """ Plays the current BlackBird instance against the previous version of
    BlackBird's neural network.

    Args:
        `model`: The Blackbird model to test
        `temp`: A float between 0 and 1 determining the exploitation
            temp for MCTS. Usually this should be close to 0.1 to ensure
            optimal move selection.
        `numTests`: An int determining the number of games to play.

    Returns:
        `wins`: The number of wins BlackBird had.
        `draws`: The number of draws BlackBird had.
        `losses`: The number of losses BlackBird had.
    """
    oldModel = model.LastVersion()

    stats = TestModels(model, oldModel, temp, numTests)
    model.Conn.PutTrainingStatistic('self', stats, model.ModelKey)
    
    del oldModel
    return stats


def TestGood(model, temp, numTests):
    """ Plays the current BlackBird instance against a standard MCTS player.

        Args:
            `model`: The Blackbird model to test
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            `wins`: The number of wins BlackBird had.
            `draws`: The number of draws BlackBird had.
            `losses`: The number of losses BlackBird had.
    """
    good = FixedMCTS(maxDepth=10, explorationRate=0.85, timeLimit=1)
    stats = TestModels(model, good, temp, numTests)
    model.Conn.PutTrainingStatistic('MCTS', stats, model.ModelKey)
    return stats


def TestModels(model1, model2, temp, numTests):
    wins = draws = losses = 0
    for _ in range(numTests):
        model1ToMove = random.choice([True, False])
        model1Player = 1 if model1ToMove else 2
        winner = None
        model1.DropRoot()
        model2.DropRoot()
        state = model1.Game()

        while winner is None:
            if model1ToMove:
                (nextState, *_) = model1.FindMove(state, temp)
            else:
                (nextState, *_) = model2.FindMove(state, temp)
            state = nextState
            model1.MoveRoot(state)
            model2.MoveRoot(state)

            model1ToMove = not model1ToMove
            winner = state.Winner()

        if winner == model1Player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses
    }


def GenerateTrainingSamples(model, nGames, temp):
    """ Generates self-play games to learn from.

        This method generates `nGames` self-play games, and stores the game
        states in a local sqlite3 database.

        Args:
            `model`: The Blackbird model to use to generate games
            `nGames`: An int determining the number of games to generate.
            `temp`: A float between 0 and 1 determining the exploration temp
                for MCTS. Usually this should be close to 1 to ensure
                high move exploration rate.

        Raises:
            ValueError: nGames was not a positive integer.
    """
    if nGames <= 0:
        raise ValueError('Use a positive integer for number of games.')

    for _ in range(nGames):
        gameHistory = []
        state = model.Game()
        lastAction = None
        winner = None
        model.DropRoot()
        while winner is None:
            (nextState, v, currentProbabilties) = model.FindMove(state, temp)
            childValues = model.Root.ChildWinRates()
            example = TrainingExample(state, 1 - v, childValues,
                                      state.EvalToString(
                                          childValues), currentProbabilties, model.Root.Priors,
                                      state.EvalToString(model.Root.Priors), state.LegalActionShape())
            state = nextState
            model.MoveRoot(state)

            winner = state.Winner(lastAction)
            gameHistory.append(example)

        example = TrainingExample(state, None, None, None,
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  state.LegalActionShape())
        gameHistory.append(example)

        for example in gameHistory:
            if winner == 0:
                example.Reward = 0
            else:
                example.Reward = 1 if example.State.Player == winner else -1

        serialized = [SerializeState(example.State, example.Probabilities, example.Reward) for example in gameHistory]
        model.Conn.PutGames(state.GameType, serialized)


def SerializeState(state, policy, reward):
    serialized = State()

    serialized.player = state.Player
    serialized.mctsEval = reward
    serialized.mctsPolicy = policy.tobytes()
    serialized.boardEncoding = state.AsInputArray().tobytes()
    serialized.boardDims = np.array(state.AsInputArray().shape, dtype=np.int8).tobytes()
    serialized.policyDims = state.LegalActionShape().tobytes()

    return serialized.SerializeToString()

def TrainWithExamples(model, batchSize, learningRate, teacher=None):
    """ Trains the neural network on provided example positions.

        Provided a list of example positions, this method will train
        BlackBird's neural network to play better. If `teacher` is provided,
        the neural network will include a cross-entropy term in the loss
        calculation so that the other network's policy is incorporated into
        the learning.

        Args:
            `model`: The Blackbird model to train
            `examples`: A list of `TrainingExample` objects which the
                neural network will learn from.
            `teacher`: An optional `BlackBird` object whose policy the
                current network will include in its loss calculation.
    """
    model.SampleValue.cache_clear()
    model.GetPriors.cache_clear()

    games = model.Conn.GetGames()
    examples = [ExampleState(game) for game in games]

    examples = np.random.choice(examples,
                                len(examples) -
                                (len(examples) % batchSize),
                                replace=False)

    for i in range(len(examples) // batchSize):
        start = i * batchSize
        batch = examples[start: start + batchSize]
        model.train(
            np.vstack([b.board for b in batch]),
            np.hstack([b.mctsEval for b in batch]),
            np.vstack([b.mctsPolicy for b in batch]),
            learningRate,
            teacher
        )

    model.Version += 1
    model.Conn.PutModel(1, model.Game.GameType, model.Name, model.Version)


class Model(MCTS, Network):
    """ Class which encapsulates MCTS powered by a neural network.

        The BlackBird class is designed to learn how to win at a board game, by
        using Monte Carlo Tree Search (MCTS) with the tree search powered by a
        neural network.
        Args:
            `game`: A GameState object which holds the rules of the game
                BlackBird is intended to learn.
            `name`: The name of the model.
            `mctsConfig` : JSON config for MCTS runtime evaluation
            `networkConfig` : JSON config for creating a new network from NetworkFactory
            `tensorflowConfig` : Configuaration for tensorflow initialization
    """

    def __init__(self, game, name, mctsConfig, networkConfig={}, tensorflowConfig={}):
        self.Conn = Connection()
        self.Game = game
        self.Name = name.split('_')[0]
        self.Version = self.Conn.SetLastModel(self.Game.GameType, self.Name)
        self.ModelKey = self.Conn.ModelKey
        self.Name += '_'.join(['', game.GameType, str(self.Version)])

        self.MCTSConfig = mctsConfig
        self.NetworkConfig = networkConfig
        self.TensorflowConfig = tensorflowConfig
        MCTS.__init__(self, **mctsConfig)

        if networkConfig != {}:
            Network.__init__(self, self.Name, NetworkFactory(networkConfig, game.LegalMoves), tensorflowConfig)
        else:
            Network.__init__(self, self.Name, tensorflowConfig=tensorflowConfig)
        self.Conn.SetLastModel(self.Game.GameType, self.Name)

    def LastVersion(self):
        return Model(self.Game, self.Name, self.MCTSConfig, self.NetworkConfig, self.TensorflowConfig)

    @functools.lru_cache(maxsize=4096)
    def SampleValue(self, state, player):
        """ Returns BlackBird's evaluation of a supplied position.

            BlackBird's network will evaluate a supplied position, from the
            perspective of `player`.

            Args:
                `state`: A GameState object which should be evaluated.
                `player`: An int representing the current player.

            Returns:
                `value`: A float between 0 and 1 holding the evaluation of the
                    position. 0 is the worst possible evaluation, 1 is the best.
        """
        value = self.getEvaluation(state.AsInputArray())
        value = (value + 1) * 0.5  # [-1, 1] -> [0, 1]
        if state.Player != player:
            value = 1 - value
        assert value >= 0, 'Value: {}'.format(value)
        return value

    @functools.lru_cache(maxsize=4096)
    def GetPriors(self, state):
        """ Returns BlackBird's policy of a supplied position.

            BlackBird's network will evaluate the policy of a supplied position.

            Args:
                `state`: A GameState object which should be evaluated.

            Returns:
                `policy`: A list of floats of size `len(state.LegalActions())` 
                    which sums to 1, representing the probabilities of selecting 
                    each legal action.
        """
        policy = self.getPolicy(state.AsInputArray()) * state.LegalActions()
        policy /= np.sum(policy)

        return policy
