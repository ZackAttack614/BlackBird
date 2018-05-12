from src.game import game
from src.blackbird import blackbird

def main():
    b = blackbird(game)

    for i in range(1, 10):
        b.selfPlay(num_games=100)
        b.train(learning_rate=0.002)
        print('Self-play score: {0}'.format(b.testNewNetwork(num_trials=50)))
        print('Random score: {0}'.format(b.testNewNetwork(against_random=True, num_trials=50)))
        print('Completed {} minibatch(es).'.format(i))

if __name__ == '__main__':
    main()
