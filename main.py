import yaml

from src.game import game
from src.blackbird import blackbird
import os

def main():
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to test runs.'
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())
        
    b = blackbird(game, parameters)

    training_parameters = parameters['selfplay']


    for i in range(1, training_parameters['epochs'] + 1):
        b.selfPlay(num_games=training_parameters['training_games'])
        b.train(learning_rate=training_parameters['learning_rate'])

        print('Self-play score: {}'.format(b.testNewNetwork(num_trials=training_parameters['selfplay_tests'])))
        print('Self-play vs low-depth score: {}'.format(
            b.testNewNetwork(against_simple=True, num_trials=training_parameters['selfplay_tests'])))
        print('Random score: {}'.format(b.testNewNetwork(against_random=True, num_trials=training_parameters['random_tests'])))
        print('Completed {} minibatch(es).'.format(i))

if __name__ == '__main__':
    main()
