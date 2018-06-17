import os
import sys
import yaml
sys.path.insert(0, './src/')

from Blackbird import BlackBird
from TicTacToe import BoardState

def main():
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to load network.'
    with open('parameters.yaml') as param_file:
        oldParams = yaml.load(param_file.read().strip())

    assert os.path.isfile('NextNetworkParams.yaml'), 'Copy parameters_template.yaml file into NextNetworkParams.yaml to bootstrap new network.'
    with open('NextNetworkParams.yaml') as param_file:
        newParams = yaml.load(param_file.read().strip())

    OldBlackbirdInstance = BlackBird(tfLog=False, loadOld=True, **oldParams)
    NewBlackbirdInstance = BlackBird(tfLog=False, loadOld=False, **newParams)

    for epoch in range(1, newParams.get('selfplay').get('epochs') + 1):
        print('Starting epoch {0}...'.format(epoch))
        nGames = newParams.get('selfplay').get('training_games')
        examples = NewBlackbirdInstance.GenerateTrainingSamples(
            nGames,
            newParams.get('mcts').get('temperature').get('exploration'))
        NewBlackbirdInstance.LearnFromExamples(examples, OldBlackbirdInstance)
        print('Finished training for this epoch!')

        NewBlackbirdInstance.saveModel()

if __name__ == '__main__':
    main()