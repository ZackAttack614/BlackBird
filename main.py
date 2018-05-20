import yaml
from time import time
import os
import json

from src.game import game
from src.blackbird import blackbird

def main():
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to test runs.'
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())

    if parameters['logging']['log_dir'] is not None and os.path.isdir(parameters['logging']['log_dir']):
        for file in os.listdir(os.path.join(os.curdir,parameters['logging']['log_dir'])):
            os.remove(os.path.join(os.curdir,parameters['logging']['log_dir'],file))
    training_parameters = parameters['selfplay']
    blackbird_instance = blackbird(game, parameters)

    for epoch in range(1, training_parameters['epochs'] + 1):
        blackbird_instance.selfPlay(num_games=training_parameters['training_games'])
        blackbird_instance.train(learning_rate=training_parameters['learning_rate'])

        selfplay_score = blackbird_instance.testNewNetwork(num_trials=training_parameters['selfplay_tests'])
        simple_score = blackbird_instance.testNewNetwork(against_simple=True, num_trials=training_parameters['selfplay_tests'])
        random_score = blackbird_instance.testNewNetwork(against_random=True, num_trials=training_parameters['random_tests'])

        print('Self-play score: {}'.format(selfplay_score))
        print('Self-play vs low-depth score: {}'.format(simple_score))
        print('Random score: {}'.format(random_score))
        print('Completed {} epoch(s).\n'.format(epoch))


if __name__ == '__main__':
    main()
