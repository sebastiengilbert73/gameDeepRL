import logging
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import rules.connect4
import simulation.simulator
import random
import architectures.connect4 as arch
import os
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
parser.add_argument('--conv1NumberOfChannels', help="The number of channels for the 1st convolution. Default: 16", type=int, default=16)
parser.add_argument('--conv2NumberOfChannels', help="The number of channels for the 2nd convolution. Default: 32", type=int, default=32)
#parser.add_argument('--hiddenSize', help="The size of the hidden layer. Default: 64", type=int, default=64)
parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.5", type=float, default=0.5)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.0001', type=float, default=0.0001)
parser.add_argument('--weightDecay', help="The weight decay. Default: 0.0001", type=float, default=0.0001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 200', type=int, default=200)
parser.add_argument('--numberOfTrainingPositions', help="The number of positions for training. Default: 100", type=int, default=100)
parser.add_argument('--numberOfSimulations', help="The number of simulations per position. Default: 100", type=int, default=100)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--modelFilepathPrefix', help="The model filepath prefix. Default: './outputs/ConvPredictorDirect_connect4_'", default='./outputs/ConvPredictorDirect_connect4_')
parser.add_argument('--faceOffNumberOfSimulations', help="When playing against a random player, the number of simulations per position. Default: 10", type=int, default=10)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

device = 'cpu'
useCuda = not args.useCpu and torch.cuda.is_available()
if useCuda:
    device = 'cuda'



class PositionStats(Dataset):
    def __init__(self, player_simulator, opponent_simulator, number_of_positions,
                 maximum_number_of_moves=42, number_of_simulations=100):
        super().__init__()
        self.player_simulator = player_simulator
        self.opponent_simulator = opponent_simulator
        self.position_stats_pairs = []
        self.authority = rules.connect4.Authority()
        self.maximum_number_of_moves = maximum_number_of_moves
        random_player = simulation.simulator.RandomSimulator()
        authority = rules.connect4.Authority()
        for positionNdx in range(number_of_positions):
            # Generate a game
            #positionsList, winner = self.player_simulator.SimulateAsymmetricGame(
            #    self.authority, self.opponent_simulator, self.maximum_number_of_moves)
            positionsList, winner = random_player.SimulateGame(authority, maximum_number_of_moves)
            # Select a position from the list
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
                    starting_position, starting_player='red')
                if sim_winner == 'yellow':
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
        return (torch.tensor(position, dtype=torch.float), torch.tensor([stats[0], stats[1], stats[2]]))


def main():
    logging.info("learn_connect4.py main()")
    authority = rules.connect4.Authority()

    opponent_simulator = simulation.simulator.RandomSimulator()
    player_simulator = simulation.simulator.RandomSimulator()

    # Create a neural network
    neural_net = arch.ConvPredictorDirect(
        conv1_number_of_channels=args.conv1NumberOfChannels,
        conv2_number_of_channels=args.conv2NumberOfChannels,
        #hidden_size=args.hiddenSize,
        dropout_ratio=args.dropoutRatio,
        final_decision_softmax_temperature=0.0,
        simulation_softmax_temperature=1.0
    ).to(device)

    logging.info("Creating training and validation datasets...")
    training_dataset = PositionStats(
        player_simulator=player_simulator,
        opponent_simulator=opponent_simulator,
        number_of_positions=args.numberOfTrainingPositions,
        maximum_number_of_moves=42,
        number_of_simulations=args.numberOfSimulations
    )
    logging.info("Finished creating training dataset")
    number_of_validation_positions = int(0.25 * args.numberOfTrainingPositions)
    validation_dataset = PositionStats(
        player_simulator=player_simulator,
        opponent_simulator=opponent_simulator,
        number_of_positions=number_of_validation_positions,
        maximum_number_of_moves=42,
        number_of_simulations=args.numberOfSimulations
    )
    logging.info("Finished creating validation dataset")

    # Create the data loaders
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.minibatchSize,
                                                 shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.minibatchSize,
                                                  shuffle=True, num_workers=2)

    # Create the optimizer
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=args.learningRate, betas=(0.5, 0.999),
                                 weight_decay=args.weightDecay)

    # Loss function
    lossFcn = torch.nn.MSELoss()

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write("epoch,trainingLoss,validationLoss\n")
    number_of_superepochs = 10
    for superepoch in range(1, number_of_superepochs + 1):
        logging.info ("****** Superepoch {} ******".format(superepoch))
        lowest_validation_loss = 1.0e9
        for epoch in range(1, args.numberOfEpochs + 1):
            #print ("epoch {}".format(epoch))
            # Set the neural network to training mode
            neural_net.train()
            loss_sum = 0
            for starting_position_tsr, training_target_stats_tsr in training_loader:
                print('.', end='', flush=True)
                starting_position_tsr, training_target_stats_tsr = starting_position_tsr.to(device), training_target_stats_tsr.to(device)
                optimizer.zero_grad()
                prediction_tsr = neural_net(starting_position_tsr)
                loss_0 = lossFcn(prediction_tsr[0], training_target_stats_tsr)
                loss_1 = lossFcn(prediction_tsr[1], training_target_stats_tsr)
                #loss_2 = lossFcn(prediction_tsr[2], training_target_stats_tsr)
                loss = 0.5 * loss_0 + 0.5 * loss_1
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            training_loss = loss_sum/training_dataset.__len__()
            print(' ', end='', flush=True)

            # Validation
            with torch.no_grad():
                neural_net.eval()
                validation_loss_sum = 0
                for starting_position_tsr, validation_target_stats_tsr in validation_loader:
                    starting_position_tsr, validation_target_stats_tsr = starting_position_tsr.to(device), validation_target_stats_tsr.to(device)
                    prediction_tsr = neural_net(starting_position_tsr)
                    loss_0 = lossFcn(prediction_tsr[0], validation_target_stats_tsr)
                    loss_1 = lossFcn(prediction_tsr[1], validation_target_stats_tsr)
                    loss = 0.5 * loss_0 + 0.5 * loss_1
                    validation_loss_sum += loss.item()
                validation_loss = validation_loss_sum/validation_dataset.__len__()
                if superepoch == number_of_superepochs and validation_loss < lowest_validation_loss:
                    lowest_validation_loss = validation_loss
                    model_filepath = args.modelFilepathPrefix + str(args.conv1NumberOfChannels) + '_' + str(
                        args.conv2NumberOfChannels) + '_' + str(
                        args.dropoutRatio) + '_' + str(superepoch) + '_' + "{:.4f}".format(validation_loss) + '.pth'
                    torch.save(neural_net.state_dict(), model_filepath)

            if epoch % 50 == 1 or epoch == args.numberOfEpochs:
                print('\n')
                logging.info("Epoch {}:   training_loss = {:.6f}   validation_loss = {:.6f}".format(epoch, training_loss, validation_loss))
            epochLossFile.write("{},{},{}\n".format((superepoch - 1) * args.numberOfEpochs + (epoch - 1), training_loss, validation_loss))
        print("")


        # Play against a random player
        logging.info("Playing against a random player...")
        #neural_net.SetSoftmaxTemperature(1.0)
        games_list, number_of_wins, number_of_draws, number_of_losses = PlayAgainstRandomPlayer(
            neural_net, 20, authority
        )
        logging.info("Superepoch {}: Against a random player: number_of_wins = {}   number_of_draws = {}   number_of_losses = {}".format(superepoch, number_of_wins, number_of_draws, number_of_losses))
        #DisplayLostGame(authority, games_list)

        # Recompute the datasets
        player_simulator = copy.deepcopy(neural_net)
        #player_simulator.SetSoftmaxTemperature(2.0)
        #opponent_simulator = simulation.simulator.RandomSimulator()  # Could be another copy of neural_net, with a different softmax temperature
        opponent_simulator = copy.deepcopy(neural_net)
        opponent_simulation_softmax_temperature = 0.3 * (number_of_superepochs - superepoch) + 1.0
        opponent_simulator.SetSimulationSoftmaxTemperature(opponent_simulation_softmax_temperature)
        logging.info("Creating training and validation datasets... opponent_simulation_softmax_temperature = {}".format(opponent_simulation_softmax_temperature))
        number_of_training_positions = args.numberOfTrainingPositions
        if superepoch == number_of_superepochs - 1:
            number_of_training_positions = 10 * args.numberOfTrainingPositions
        number_of_validation_positions = int(0.25 * number_of_training_positions)
        training_dataset = PositionStats(
            player_simulator=player_simulator,
            opponent_simulator=opponent_simulator,
            number_of_positions=number_of_training_positions,
            maximum_number_of_moves=42,
            number_of_simulations=args.numberOfSimulations
        )
        logging.info("Finished creating training dataset")
        validation_dataset = PositionStats(
            player_simulator=player_simulator,
            opponent_simulator=opponent_simulator,
            number_of_positions=number_of_validation_positions,
            maximum_number_of_moves=42,
            number_of_simulations=args.numberOfSimulations
        )
        logging.info("Finished creating validation dataset")
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.minibatchSize,
                                                      shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.minibatchSize,
                                                        shuffle=True, num_workers=2)

        model_filepath = args.modelFilepathPrefix + str(args.conv1NumberOfChannels) + '_' + str(args.conv2NumberOfChannels) + '_' + str(args.dropoutRatio) + '_' + str(superepoch) + '.pth'
        torch.save(neural_net.state_dict(), model_filepath)


def PlayAgainstRandomPlayer(player_simulator, number_of_games, authority):
    number_of_wins = 0
    number_of_draws = 0
    number_of_losses = 0
    games_list = []
    random_simulator = simulation.simulator.RandomSimulator()
    for gameNdx in range(number_of_games):
        positionsList = None
        winner = None
        if gameNdx % 2 == 0:
            positionsList, winner = player_simulator.SimulateAsymmetricGameMonteCarlo(
                authority=authority,
                other_player_simulator=random_simulator,
                maximum_number_of_moves=42,
                number_of_simulations=args.faceOffNumberOfSimulations,
                starting_position=None,
                starting_player='yellow'
            )
        else:
            positionsList, winner = random_simulator.SimulateAsymmetricGameMonteCarlo(
                authority=authority,
                other_player_simulator=player_simulator,
                maximum_number_of_moves=42,
                number_of_simulations=args.faceOffNumberOfSimulations,
                starting_position=None,
                starting_player='red'
            )
        games_list.append((positionsList, winner))
        display_character = '-'
        if winner == 'yellow':
            number_of_wins += 1
            display_character = 'yellow'
        elif winner == 'red':
            number_of_losses += 1
            display_character = 'red'
        else:
            number_of_draws += 1
        print('{}, '.format(display_character), end='', flush=True)
    print()
    return games_list, number_of_wins, number_of_draws, number_of_losses

def DisplayLostGame(authority, games_list):
    for game in games_list:
        if game[1] == 'red':
            logging.info("Lost game:")
            authority.DisplayGame(game[0])
            break

if __name__ == '__main__':
    main()