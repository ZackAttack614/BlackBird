import os
import sys
import yaml
sys.path.insert(0, './src/')

import Blackbird
from TicTacToe import BoardState


def APITest():
    if not os.path.isfile('parameters.yaml'):
        raise IOError('Copy parameters_template.yaml into parameters.yaml')
    with open('parameters.yaml') as param_file:
        parameters = yaml.safe_load(param_file)

    model = Blackbird.Model(BoardState, parameters['name'], parameters.get(
        'mcts'), parameters.get('network'), parameters.get('tensorflow'))

    Blackbird.GenerateTrainingSamples(model,
                                    10,
                                    parameters.get('mcts').get('temperature').get('exploration'))
    Blackbird.TrainWithExamples(model, batchSize=10, learningRate=0.01)

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
