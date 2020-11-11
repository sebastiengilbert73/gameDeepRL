import logging
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import rules.tictactoe
import simulation.simulator
import random

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './'", default='./')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

class PositionStats(Dataset):
    def __init__(self, player_simulator, opponent_simulator, number_of_positions,
                 start_simulation_at_index=-3, maximum_number_of_moves=9, number_of_simulations=100):
        super().__init__()
        self.player_simulator = player_simulator
        self.opponent_simulator = opponent_simulator
        self.position_stats_pairs = []
        self.authority = rules.tictactoe.Authority()
        self.maximum_number_of_moves = maximum_number_of_moves
        for positionNdx in range(number_of_positions):
            # Generate a game
            positionsList, winner = self.player_simulator.SimulateAsymmetricGame(
                self.authority, self.opponent_simulator, self.maximum_number_of_moves)
            # Select a position from the list
            """startNdx = start_simulation_at_index
            if len(positionsList) < -startNdx:
                startNdx = 0
            if startNdx < 0:
                startNdx += len(positionsList)
            """
            startNdx = random.randint(0, len(positionsList) - 2)

            positionsList = positionsList[: startNdx + 1]
            if len(positionsList) % 2 == 0:
                starting_position = positionsList[-1]
            else:
                starting_position = self.authority.SwapPositions(positionsList[-1])
            # Run simulations
            number_of_wins = 0
            number_of_draws = 0
            number_of_losses = 0
            for simulationNdx in range(number_of_simulations):
                sim_positions, sim_winner = self.opponent_simulator.SimulateAsymmetricGame(
                    self.authority, self.player_simulator, self.maximum_number_of_moves,
                    starting_position, starting_player='O')
                if sim_winner == 'X':
                    number_of_wins += 1
                elif sim_winner == 'draw':
                    number_of_draws += 1
                else:
                    number_of_losses += 1
            self.position_stats_pairs.append((starting_position,
                                              (number_of_wins/number_of_simulations, number_of_draws/number_of_simulations, number_of_losses/number_of_simulations)))

    def __len__(self):
        return len(self.position_stats_pairs)

    def __getitem__(self, idx):
        position, stats = self.position_stats_pairs[idx]
        return (torch.tensor(position), torch.tensor([stats[0], stats[1], stats[2]]))


def main():
    logging.info("learn_tictactoe.py main()")
    authority = rules.tictactoe.Authority()
    player_simulator = simulation.simulator.RandomSimulator()
    opponent_simulator = simulation.simulator.RandomSimulator()
    position_stats = PositionStats(player_simulator, opponent_simulator, number_of_positions=10,
                                   start_simulation_at_index=0, maximum_number_of_moves=9,
                                   number_of_simulations=100)
    for index in range(position_stats.__len__()):
        (position, stats) = position_stats[index]
        print("{}: {}".format(position, stats))

if __name__ == '__main__':
    main()