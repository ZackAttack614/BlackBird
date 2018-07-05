import os
import sys
import yaml
sys.path.insert(0, './src/')

from Blackbird import BlackBird
from TicTacToe import BoardState


def main():
    if not os.path.isfile('parameters.yaml'):
        raise IOError('Copy parameters_template.yaml into parameters.yaml')
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())

    numEpochs = parameters.get('selfplay').get('epochs')
    BlackbirdInstance = BlackBird(BoardState, tfLog=True,
                                  **parameters)

    for epoch in range(1, numEpochs + 1):
        print('Starting epoch {0}...'.format(epoch))
        examples = BlackbirdInstance.GenerateTrainingSamples(
            parameters.get('selfplay').get('training_games'),
            parameters.get('mcts').get('temperature').get('exploration'))
        BlackbirdInstance.LearnFromExamples(examples)
        print('Finished training for this epoch!')

        (wins, draws, losses) = BlackbirdInstance.TestRandom(
            parameters.get('mcts').get('temperature').get('exploitation'),
            parameters.get('selfplay').get('random_tests'))
        print('Against a random player:')
        print('Wins = {0}'.format(wins))
        print('Draws = {0}'.format(draws))
        print('Losses = {0}'.format(losses))

        (wins, draws, losses) = BlackbirdInstance.TestPrevious(
            parameters.get('mcts').get('temperature').get('exploitation'),
            parameters.get('selfplay').get('selfplay_tests'))
        print('Against the last best player:')
        print('Wins = {0}'.format(wins))
        print('Draws = {0}'.format(draws))
        print('Losses = {0}'.format(losses))

        if wins > losses:
            BlackbirdInstance.saveModel()

        (wins, draws, losses) = BlackbirdInstance.TestGood(
            parameters.get('mcts').get('temperature').get('exploitation'),
            parameters.get('selfplay').get('selfplay_tests'))
        print('Against a good player:')
        print('Wins = {0}'.format(wins))
        print('Draws = {0}'.format(draws))
        print('Losses = {0}'.format(losses))

        print('\n')
    del BlackbirdInstance
    print('Done!')


if __name__ == '__main__':
    main()
