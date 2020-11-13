import torch
import numpy as np
import simulation.simulator
import math
import random


class ConvPredictor(torch.nn.Module, simulation.simulator.Simulator):
    def __init__(self, conv1_number_of_channels, conv2_number_of_channels,
                 hidden_size, dropout_ratio=0.5, soft_max_temperature=1.0):
        super(ConvPredictor, self).__init__()
        self.conv1 = torch.nn.Conv3d(2, conv1_number_of_channels, (1, 2, 2))
        self.conv2 = torch.nn.Conv3d(conv1_number_of_channels, conv2_number_of_channels, (1, 2, 2))
        self.fc1 = torch.nn.Linear(conv2_number_of_channels, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 3)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.conv2_number_of_channels = conv2_number_of_channels
        self.soft_max_temperature = soft_max_temperature

    def forward(self, x):
        # x.shape = (N, 2, 1, 3, 3)
        activation1 = torch.nn.functional.relu(self.conv1(x))
        # activation1.shape = (N, c1, 1, 2, 2)
        activation2 = torch.nn.functional.relu(self.conv2(activation1))
        # activation2.shape = (N, c2, 1, 1, 1)
        activation2 = activation2.view(-1, self.conv2_number_of_channels)
        hidden = torch.nn.functional.relu(self.fc1(activation2))
        # hidden.shape = (N, h)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        # output.shape = (N, 3)
        return torch.nn.functional.softmax(output, dim=1)

    def ChooseMoveCoordinates(self, authority, position, player):
        legal_move_coordinates = authority.LegalMoveCoordinates(position, player)
        move_to_choice_probability = {}
        other_player = authority.OtherPlayer(player)
        for move_coordinates in legal_move_coordinates:
            resulting_position, winner = authority.MoveWithMoveArrayCoordinates(position, player, move_coordinates)
            move_coordinates = tuple(move_coordinates)  # To make it hashable
            if winner == player:
                return move_coordinates
            elif winner == other_player:
                move_to_choice_probability[move_coordinates] = -1
            elif winner == 'draw':
                move_to_choice_probability[move_coordinates] = 0
            else:
                resulting_position_tsr = torch.tensor(resulting_position, dtype=torch.float).unsqueeze(0)
                print ("ConvPredictor.ChooseMoveCoordinates(): resulting_position_tsr = {}".format(resulting_position_tsr))
                predictionTsr = self.forward(resulting_position_tsr).squeeze(0)
                expected_value = predictionTsr[0].item() - predictionTsr[2].item()
                move_to_choice_probability[move_coordinates] = expected_value

        print("ConvPredictor.ChooseMoveCoordinates(): Before normalization, move_to_choice_probability = \n{}".format(
            move_to_choice_probability))
        # Normalize
        sum = 0
        for move, expected_value in move_to_choice_probability.items():
            sum += math.exp(expected_value/self.soft_max_temperature)
        for move, expected_value in move_to_choice_probability.items():
            move_to_choice_probability[move] = math.exp(expected_value)/sum
        print("ConvPredictor.ChooseMoveCoordinates(): move_to_choice_probability = \n{}".format(move_to_choice_probability))

        # Draw a random number
        random_draw = random.random()
        running_sum = 0
        for move, probability in move_to_choice_probability.items():
            running_sum += probability
            if running_sum >= random_draw:
                return move
        raise RuntimeError("ConvPredictor.ChooseMoveCoordinates(): Summed the probabilities without reaching the random number {}".format(random_draw))
