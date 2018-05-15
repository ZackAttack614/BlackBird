import yaml

from src.game import game
from src.blackbird import blackbird

def main():
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())

    b = blackbird(game, parameters)

    training_parameters = parameters['selfplay']

    for i in range(1, training_parameters['minibatches'] + 1):
        b.selfPlay(num_games=training_parameters['selfplay_games'])
        b.train(learning_rate=training_parameters['learning_rate'])

        print('Self-play score: {}'.format(b.testNewNetwork(num_trials=training_parameters['selfplay_tests'])))
        print('Self-play vs low-depth score: {}'.format(
            b.testNewNetwork(against_simple=True, num_trials=training_parameters['selfplay_tests'])))
        print('Random score: {}'.format(b.testNewNetwork(against_random=True, num_trials=training_parameters['random_tests'])))
        print('Completed {} minibatch(es).'.format(i))

if __name__ == '__main__':
    main()
