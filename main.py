import os
import sys
import json
sys.path.insert(0, './src/')

import Blackbird
from TicTacToe import BoardState


def APITest():
    if not os.path.isfile('parameters.json'):
        raise IOError('Copy parameters_template.json into parameters.json')
    with open('parameters.json') as param_file:
        parameters = json.load(param_file)

    model = Blackbird.Model(BoardState, parameters['name'], parameters.get(
        'mcts'), parameters.get('network'), parameters.get('tensorflow'))

    examples = Blackbird.GenerateTrainingSamples(model,
                                                 10,
                                                 parameters.get('mcts').get('temperature').get('exploration'))
    Blackbird.TrainWithExamples(
        model, examples, batchSize=10, learningRate=0.01)

    print('Against a random player:')
    print(Blackbird.TestRandom(model,
                               parameters.get('mcts').get(
                                   'temperature').get('exploitation'),
                               10))

    print('Against the last best player:')
    print(Blackbird.TestPrevious(model,
                                 parameters.get('mcts').get(
                                     'temperature').get('exploitation'),
                                 10))

    print('Against a good player:')
    print(Blackbird.TestGood(model,
                             parameters.get('mcts').get(
                                 'temperature').get('exploitation'),
                             10))

    print('\n')
    del model


if __name__ == '__main__':
    APITest()
