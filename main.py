from pymongo import MongoClient
import yaml
import os

from src.game import game
from src.blackbird import blackbird

def main():
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to test runs.'
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())
        
    training_parameters = parameters['selfplay']
    blackbird_instance = blackbird(game, parameters)

    if parameters['logging']['remote_logging']:
        client = MongoClient(parameters['logging']['log_server'],5000)
        db = client.blackbird_test_games
        test_results = db.test_results
        test_results.delete_many({"limit":False})

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

        if parameters['logging']['remote_logging']:
            test_results.insert_one({
                'blocks':parameters['network']['blocks'],
                'filters':parameters['network']['filters'],
                'playout_depth':parameters['mcts']['playouts'],
                'selfplay_against_best': selfplay_score,
                'selfplay_against_simple': simple_score,
                'play_against_random': random_score
            })

if __name__ == '__main__':
    main()
