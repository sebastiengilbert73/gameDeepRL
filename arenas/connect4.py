import logging
import argparse
import rules.connect4
import simulation.simulator
import random
import architectures.connect4 as arch
import torch
import sys
import math

parser = argparse.ArgumentParser()
parser.add_argument('NeuralNetworkFilepath', help="The filepath of the neural network")
parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
parser.add_argument('--conv1NumberOfChannels', help="The number of channels for the 1st convolution. Default: 16", type=int, default=16)
parser.add_argument('--conv2NumberOfChannels', help="The number of channels for the 2nd convolution. Default: 32", type=int, default=32)
parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.5", type=float, default=0.5)
parser.add_argument('--finalDecisionSoftmaxTemperature', help="The softmax temperature for the final decision. Default: 0.0", type=float, default=0.0)
parser.add_argument('--simulationSoftmaxTemperature', help="The softmax temperature for the simulation quick decisions. Default: 1.0", type=float, default=1.0)
parser.add_argument('--numberOfSimulations', help="The number of simulations per position. Default: 30", type=int, default=30)
parser.add_argument('--neuralNetworkPlaysFirst', action='store_true')
parser.add_argument('--displayLegalMoveStatistics', action='store_true')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

device = 'cpu'
useCuda = not args.useCpu and torch.cuda.is_available()
if useCuda:
    device = 'cuda'


def main():
    logging.info("arenas/connect4.py main()\tdevice = {}".format(device))

    # Load the neural network
    """neural_net = arch.ConvPredictorDirect(
        conv1_number_of_channels=args.conv1NumberOfChannels,
        conv2_number_of_channels=args.conv2NumberOfChannels,
        dropout_ratio=args.dropoutRatio,
        final_decision_softmax_temperature=args.finalDecisionSoftmaxTemperature,
        simulation_softmax_temperature=args.simulationSoftmaxTemperature
    ).to(device)
    """
    neural_net = arch.ConvPredictor2Scales(
        16, 32, 16, 0.5
    )
    try:
        neural_net.load_state_dict(torch.load(args.NeuralNetworkFilepath))
    except:
        logging.error("Could not load file '{}'".format(args.NeuralNetworkFilepath))
        sys.exit()

    neural_net.eval()
    logging.info("Neural network loaded")

    authority = rules.connect4.Authority()
    players = authority.PlayersList()
    human_player = players[0]
    neural_net_player = players[1]
    if args.neuralNetworkPlaysFirst:
        human_player = players[1]
        neural_net_player = players[0]

    position = authority.InitialPosition()
    authority.Display(position)
    current_player = players[0]
    winner = None
    while winner is None:
        if current_player == human_player:
            dropColumn = int(input("Column: "))
            position, winner = authority.MoveWithColumn(position, current_player, dropColumn)
        else:
            legal_move_to_statistics_list = neural_net.LegalMoveStatistics(authority, 42, args.numberOfSimulations, position, current_player)
            if args.displayLegalMoveStatistics:
                print(legal_move_to_statistics_list)
            chosen_move = FinalDecision(legal_move_to_statistics_list, neural_net)
            position, winner = authority.MoveWithMoveArrayCoordinates(position, current_player, chosen_move)
        authority.Display(position)
        current_player = authority.OtherPlayer(current_player)
    print ("winner: {}".format(winner))


def FinalDecision(legal_move_to_statistics_list, neural_network):
    if neural_network.final_decision_softmax_temperature <= 0:  # Hard max
        highest_expected_value = -2.0
        chosen_move_coordinates = []
        for (move, stats) in legal_move_to_statistics_list:
            expected_value = (stats[0] - stats[2]) / (stats[0] + stats[1] + stats[2])
            if expected_value > highest_expected_value:
                highest_expected_value = expected_value
                chosen_move_coordinates = [move]
            elif expected_value == highest_expected_value:
                chosen_move_coordinates.append(move)
        if len(chosen_move_coordinates) == 0:
            raise ValueError(
                "arenas.connect4.FinalDecision(): chosen_move_coordinates is empty. legal_move_to_statistics_list = {}".format(
                    legal_move_to_statistics_list))
        return random.choice(chosen_move_coordinates)

    # Softmax
    move_to_expected_value_dic = {move: (stats[0] - stats[2]) / (stats[0] + stats[1] + stats[2]) for
                                  (move, stats) in legal_move_to_statistics_list}
    # Normalize
    sum = 0
    move_to_choice_probability = {}
    for move, expected_value in move_to_expected_value_dic.items():
        sum += math.exp(expected_value / neural_network.final_decision_softmax_temperature)
    if sum > 0:
        for move, expected_value in move_to_expected_value_dic.items():
            move_to_choice_probability[move] = (math.exp(
                expected_value / neural_network.final_decision_softmax_temperature)) / sum
    else:
        raise ValueError(
            "arenas.connect4.FinalDecision(): sum of exponentials ({}) is not > 0. move_to_expected_value_dic = {}".format(
                sum, move_to_expected_value_dic))

    # Draw a random number
    random_draw = random.random()
    running_sum = 0
    for move, probability in move_to_choice_probability.items():
        running_sum += probability
        if running_sum >= random_draw:
            return move
    raise RuntimeError(
        "arenas.connect4.FinalDecision(): Summed the probabilities without reaching the random number {}. move_to_choice_probability = {}".format(
            random_draw, move_to_choice_probability))

if __name__ == '__main__':
    main()