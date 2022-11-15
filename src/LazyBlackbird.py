from DynamicLazyMCTS import DynamicLazyMCTS as LazyMCTS
from RandomLazyMCTS import RandomLazyMCTS
from FixedLazyMCTS import FixedLazyMCTS
from Network import Network
from NetworkFactory import NetworkFactory
from DataManager import Connection
from proto.state_pb2 import State
from collections import defaultdict

import functools
import random
import yaml
import numpy as np

import tracemalloc
tracemalloc.start()

import time

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class ExampleState(object):
    """ Class which centralizes the data structure of a game state.

        `GameState` objects have many properties, but only a few of them are
        relevant to training. `ExampleState` provides an interface on top of
        protocol buffers for reading and storing game state data.

        Attributes:
            `MctsPolicy`: A numpy array which holds the policy generated from
                applying MCTS.
            `MctsEval`: A float between -1 and 1 representing the evaluation
                that the MCTS computed.
            `Board`: A numpy array which holds the input state for a board
                state. In general, this is the game state, as well as layers for
                historical positions, current turn, current player, etc.
            `Player`: An optional integer, representing the current player.
    """
    def __init__(self, evaluation, policy, board, player=None):
        self.MctsPolicy = policy
        self.MctsEval = evaluation
        self.Board = board
        self.Player = player

    @classmethod
    def FromSerialized(cls, serialState):
        """ Transforms a protobuf bytecode string to an `ExampleState` object.

            Args:
                serialState: A protobuf bytecode string holding `GameState`
                    info.

            Returns:
                An `ExampleState` object holding the relevant deserialized data.
        """
        state = State()
        state.ParseFromString(serialState)
        boardDims = np.frombuffer(state.boardDims, dtype=np.int8)
        policyDims = np.frombuffer(state.policyDims, dtype=np.int16)

        mctsEval = state.mctsEval,
        mctsPolicy = np.frombuffer(state.mctsPolicy,
            dtype=np.float).reshape(policyDims)
        board = np.frombuffer(state.boardEncoding,
            dtype=np.int8).reshape(boardDims)

        return cls(mctsEval, mctsPolicy, board)

    def SerializeState(self):
        """ Returns the protobuf bytecode serialization of the `ExampleState`.
        """
        serialized = State()

        serialized.mctsEval = self.MctsEval
        serialized.mctsPolicy = self.MctsPolicy.tobytes()
        serialized.boardEncoding = self.Board.tobytes()
        serialized.boardDims = np.array(self.Board.shape,
            dtype=np.int8).tobytes()
        serialized.policyDims = np.array(self.MctsPolicy.shape,
            dtype=np.int16).tobytes()

        return serialized.SerializeToString()


def TestRandom(model, temp, numTests):
    """ Plays the current BlackBird instance against an opponent making
        random moves.

        Game statistics are logged in the local `data/blackbird.db` database.

        Args:
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            A dictionary holding:
                - `wins`: The number of wins `model` had.
                - `draws`: The number of draws `model` had.
                - `losses`: The number of losses `model` had.
    """
    resultMap = {1:'wins', 0:'draws', -1:'losses'}
    stats = defaultdict(int)

    for _ in range(numTests):
        result = TestModels(model, RandomLazyMCTS(), temp, numTests=1)
        stats[resultMap.get(result, 'indeterminant')] += 1
        model.Conn.PutTrainingStatistic(result, model.Name,
            model.Version, 'RANDOM')

    return stats


def TestPrevious(model, temp, numTests):
    """ Plays the current BlackBird instance against the previous version of
        BlackBird's neural network.

        Game statistics are logged in the local `data/blackbird.db` database.

        Args:
            `model`: The Blackbird model to test
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            A dictionary holding:
                - `wins`: The number of wins `model` had.
                - `draws`: The number of draws `model` had.
                - `losses`: The number of losses `model` had.
    """
    oldModel = model.LastVersion()
    resultMap = {1:'wins', 0:'draws', -1:'losses'}
    stats = defaultdict(int)

    for _ in range(numTests):
        result = TestModels(model, oldModel, temp, numTests=1)
        stats[resultMap.get(result, 'indeterminant')] += 1
        model.Conn.PutTrainingStatistic(result, model.Name, model.Version,
            oldModel.Name, oldModel.Version)

    return stats


def TestGood(model, temp, numTests):
    """ Plays the current BlackBird instance against a standard MCTS player.

        Game statistics are logged in the local `data/blackbird.db` database.

        Args:
            `model`: The Blackbird model to test
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            A dictionary holding:
                - `wins`: The number of wins `model` had.
                - `draws`: The number of draws `model` had.
                - `losses`: The number of losses `model` had.
    """
    good = FixedLazyMCTS(maxDepth=10, explorationRate=0.05, timeLimit=1)
    resultMap = {1:'wins', 0:'draws', -1:'losses'}
    stats = defaultdict(int)
    
    for _ in range(numTests):
        result = TestModels(model, good, temp, numTests=1)
        stats[resultMap.get(result, 'indeterminant')] += 1
        model.Conn.PutTrainingStatistic(result, model.Name, model.Version,
            'MCTS')

    return stats


def TestModels(model1, model2, temp, numTests):
    """ Base function for playing a BlackBird instance against another model.

        Args:
            `model1`: The Blackbird model to test.
            `model2`: The model to play against.
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            An integer representing a win (1), draw (0), or loss (-1)
    """
    for _ in range(numTests):
        print(f'Playing game {_}\n')
        model1ToMove = random.choice([True, False])
        model1Player = 1 if model1ToMove else 2
        winner = None
        model1.DropRoot()
        model2.DropRoot()
        state = model1.Game()

        while winner is None:
            print(state)
            if model1ToMove:
                (nextState, *_) = model1.FindMove(state, temp)
            else:
                (nextState, *_) = model2.FindMove(state, temp)

            model1ToMove = not model1ToMove if state.Player != nextState.Player else model1ToMove

            state = nextState
            model1.MoveRoot(state)
            model2.MoveRoot(state)

            winner = state.Winner()

        print(state)
        print(winner)

        if winner == model1Player:
            return 1
        elif winner == 0:
            return 0
        else:
            return -1


def GenerateTrainingSamples(model, nGames, temp):
    import cProfile, pstats
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
    start = time.time()
    for i in range(nGames):
        profiler = cProfile.Profile()
        profiler.enable()
        print(f'Starting training sample {i} at time {time.time()-start}')
        gameHistory = []
        state = model.Game()
        lastAction = None
        winner = None
        model.DropRoot()
        while winner is None:
            (nextState, v, currentProbabilties) = model.FindMove(state, temp)
            example = ExampleState(1 - v, currentProbabilties,
                state.AsInputArray(), player=state.Player)
            state = nextState
            # print(f'blackbird state: {state}')
            model.MoveRoot(state)
            # print(f'moved root: {model.Game().Board}')
            winner = state.Winner(lastAction)
            gameHistory.append(example)
        print(f'\nWinner: {winner}\n')

        example = ExampleState(None, np.zeros([len(currentProbabilties)]),
            state.AsInputArray(), player=state.Player)
        gameHistory.append(example)

        for example in gameHistory:
            if winner == 0:
                example.MctsEval = 0
            else:
                example.MctsEval = 1 if example.Player == winner else -1

        serialized = [example.SerializeState() for example in gameHistory]
        model.Conn.PutGames(model.Name, model.Version, state.GameType,
            serialized)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats('profile.log')


def TrainWithExamples(model, batchSize, learningRate, epochs=1, teacher=None,
                      model_override=None, version_override=None):
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

    states = model.Conn.GetGames(model_override if model_override is not None else model.Name,
                                 version_override if version_override is not None else model.Version)
    examples = [ExampleState.FromSerialized(state) for state in states]

    examples = np.random.choice(examples,
                                len(examples) -
                                (len(examples) % batchSize),
                                replace=False)

    for i in range(len(examples) // batchSize):
        start = i * batchSize
        batch = examples[start: start + batchSize]
        model.train(
            np.vstack([b.Board for b in batch]),
            np.hstack([b.MctsEval for b in batch]),
            np.vstack([b.MctsPolicy for b in batch]),
            learningRate,
            teacher
        )

    model.Version += 1
    model.Conn.PutModel(model.Game.GameType, model.Name, model.Version)


class Model(LazyMCTS, Network):
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
        self.Name = name
        self.Version = self.Conn.GetLastVersion(self.Game.GameType, self.Name)
        self._saveName = self.Name + '_' + str(self.Version)

        self.MCTSConfig = mctsConfig
        self.NetworkConfig = networkConfig
        self.TensorflowConfig = tensorflowConfig
        LazyMCTS.__init__(self, **mctsConfig)

        if networkConfig != {}:
            Network.__init__(self, self._saveName, NetworkFactory(networkConfig, game.LegalMoves), tensorflowConfig)
        else:
            Network.__init__(self, self._saveName, tensorflowConfig=tensorflowConfig)

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
        legalActions = state.LegalActions()
        policy = self.getPolicy(state.AsInputArray()) * legalActions
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        reducedPolicy = np.delete(policy, np.where(legalActions == 0)[0])
        reducedPolicy /= np.sum(reducedPolicy)
        del policy

        return reducedPolicy, np.where(legalActions != 0)[0]