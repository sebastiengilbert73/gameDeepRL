import torch
import numpy as np
import simulation.simulator
import math
import random


class AuxiliaryRegressorConv(torch.nn.Module):
    def __init__(self, input_size_CDHW, output_size, dropout_ratio):
        super(AuxiliaryRegressorConv, self).__init__()
        self.number_of_inputs = input_size_CDHW[0] * input_size_CDHW[1] * input_size_CDHW[2] * input_size_CDHW[3]
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.linear = torch.nn.Linear(self.number_of_inputs, output_size)

    def forward(self, x):
        # x.shape = (N, C, D, H, W)
        activation = x.view(-1, self.number_of_inputs)
        activation = self.dropout(activation)
        outputTsr = torch.clip(self.linear(activation), 0, 1)  # outputTsr.shape = (N, output_size)
        outputTsr = torch.nn.functional.normalize(outputTsr, p=1, dim=1)
        return outputTsr

class AuxiliaryRegressorLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout_ratio):
        super(AuxiliaryRegressorLinear, self).__init__()
        self.number_of_inputs = input_size
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.linear = torch.nn.Linear(self.number_of_inputs, output_size)

    def forward(self, x):
        # x.shape = (N, self.number_of_inputs)
        activation = self.dropout(x)
        outputTsr = torch.clip(self.linear(activation), 0, 1)  # outputTsr.shape = (N, output_size)
        outputTsr = torch.nn.functional.normalize(outputTsr, p=1, dim=1)
        return outputTsr

class ConvPredictor(torch.nn.Module, simulation.simulator.Simulator):
    def __init__(self, conv1_number_of_channels, conv2_number_of_channels,
                 hidden_size, dropout_ratio=0.5, final_decision_softmax_temperature=0.0, simulation_softmax_temperature=1.0):
        super(ConvPredictor, self).__init__()
        self.conv1 = torch.nn.Conv3d(2, conv1_number_of_channels, (1, 2, 2))
        self.conv2 = torch.nn.Conv3d(conv1_number_of_channels, conv2_number_of_channels, (1, 2, 2))
        self.dropout3d = torch.nn.Dropout3d(p=dropout_ratio)
        self.fc1 = torch.nn.Linear(conv2_number_of_channels, hidden_size)
        #self.fc2 = torch.nn.Linear(hidden_size, 3)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.conv1_number_of_channels = conv1_number_of_channels
        self.conv2_number_of_channels = conv2_number_of_channels
        self.hidden_size = hidden_size

        self.pred1 = AuxiliaryRegressorConv((self.conv1_number_of_channels, 1, 2, 2), 3, dropout_ratio)
        self.pred2 = AuxiliaryRegressorLinear(self.conv2_number_of_channels, 3, dropout_ratio)
        self.pred3 = AuxiliaryRegressorLinear(self.hidden_size, 3, dropout_ratio)

        self.final_decision_softmax_temperature = final_decision_softmax_temperature
        self.simulation_softmax_temperature = simulation_softmax_temperature

    def forward(self, x):
        # x.shape = (N, 2, 1, 3, 3)
        activation1 = torch.nn.functional.relu(self.conv1(x))
        # activation1.shape = (N, c1, 1, 2, 2)
        activation1 = self.dropout3d(activation1)
        activation2 = torch.nn.functional.relu(self.conv2(activation1))
        # activation2.shape = (N, c2, 1, 1, 1)
        activation2 = activation2.view(-1, self.conv2_number_of_channels)
        # activation2.shape = (N, c2)
        hidden = torch.nn.functional.relu(self.fc1(activation2))
        # hidden.shape = (N, h)

        #hidden = self.dropout(hidden)
        #output = self.fc2(hidden)
        # output.shape = (N, 3)
        #return torch.nn.functional.softmax(output, dim=1)

        regression1 = self.pred1(activation1)
        regression2 = self.pred2(activation2)
        regression3 = self.pred3(hidden)
        return (regression1, regression2, regression3)


    def ChooseMoveCoordinatesQuick(self, authority, position, player):
        legal_move_coordinates = authority.LegalMoveCoordinates(position, player)
        move_to_choice_probability = {}
        other_player = authority.OtherPlayer(player)
        for move_coordinates in legal_move_coordinates:
            #print ("ConvPredictor.ChooseMoveCoordinates(): move_coordinates = {}".format(move_coordinates))
            resulting_position, winner = authority.MoveWithMoveArrayCoordinates(position, player, move_coordinates)
            if player == 'O':
                resulting_position = authority.SwapPositions(resulting_position)
            move_coordinates = tuple(move_coordinates)  # To make it hashable
            if winner == player:
                return move_coordinates
            elif winner == other_player:
                move_to_choice_probability[move_coordinates] = -1
            elif winner == 'draw':
                move_to_choice_probability[move_coordinates] = 0
            else:
                resulting_position_tsr = torch.tensor(resulting_position, dtype=torch.float).unsqueeze(0)
                # print ("ConvPredictor.ChooseMoveCoordinates(): resulting_position_tsr = {}".format(resulting_position_tsr))
                predictionTsr = self.forward(resulting_position_tsr)[2].squeeze(0)
                expected_value = predictionTsr[0].item() - predictionTsr[2].item()
                move_to_choice_probability[move_coordinates] = expected_value
            #print("ConvPredictor.ChooseMoveCoordinates(): expected_value = {}".format(move_to_choice_probability[move_coordinates]))

        softmax_temperature = self.simulation_softmax_temperature  # Quick decision

        if softmax_temperature <= 0:  # Hard max
            highest_expected_value = -2.0
            chosen_move_coordinates = []
            for move, expected_value in move_to_choice_probability.items():
                if expected_value > highest_expected_value:
                    highest_expected_value = expected_value
                    chosen_move_coordinates = [move]
                elif expected_value == highest_expected_value:
                    chosen_move_coordinates.append(move)
            return random.choice(chosen_move_coordinates)

        # print("ConvPredictor.ChooseMoveCoordinates(): Before normalization, move_to_choice_probability = \n{}".format(move_to_choice_probability))
        # Normalize
        sum = 0
        for move, expected_value in move_to_choice_probability.items():
            sum += math.exp(expected_value/softmax_temperature)

        for move, expected_value in move_to_choice_probability.items():
            move_to_choice_probability[move] = (math.exp(expected_value/softmax_temperature) )/sum

        # Draw a random number
        random_draw = random.random()
        running_sum = 0
        for move, probability in move_to_choice_probability.items():
            running_sum += probability
            if running_sum >= random_draw:
                return move
        raise RuntimeError("ConvPredictor.ChooseMoveCoordinates(): Summed the probabilities without reaching the random number {}".format(random_draw))



    def SetSimulationSoftmaxTemperature(self, temperature):
        self.simulation_softmax_temperature = temperature


class ConvPredictorDirect(torch.nn.Module, simulation.simulator.Simulator):
    def __init__(self, conv1_number_of_channels, conv2_number_of_channels,
                 dropout_ratio=0.5, final_decision_softmax_temperature=0.0, simulation_softmax_temperature=1.0):
        super(ConvPredictorDirect, self).__init__()
        self.conv1 = torch.nn.Conv3d(2, conv1_number_of_channels, (1, 2, 2))
        self.conv2 = torch.nn.Conv3d(conv1_number_of_channels, conv2_number_of_channels, (1, 2, 2))
        self.dropout3d = torch.nn.Dropout3d(p=dropout_ratio)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.conv1_number_of_channels = conv1_number_of_channels
        self.conv2_number_of_channels = conv2_number_of_channels

        self.pred1 = AuxiliaryRegressorConv((self.conv1_number_of_channels, 1, 2, 2), 3, dropout_ratio)
        self.pred2 = AuxiliaryRegressorLinear(self.conv2_number_of_channels, 3, dropout_ratio)

        self.final_decision_softmax_temperature = final_decision_softmax_temperature
        self.simulation_softmax_temperature = simulation_softmax_temperature

    def forward(self, x):
        # x.shape = (N, 2, 1, 3, 3)
        activation1 = torch.nn.functional.relu(self.conv1(x))
        # activation1.shape = (N, c1, 1, 2, 2)
        activation1 = self.dropout3d(activation1)
        activation2 = torch.nn.functional.relu(self.conv2(activation1))
        # activation2.shape = (N, c2, 1, 1, 1)
        activation2 = activation2.view(-1, self.conv2_number_of_channels)
        # activation2.shape = (N, c2)

        regression1 = self.pred1(activation1)
        regression2 = self.pred2(activation2)
        return (regression1, regression2)


    def ChooseMoveCoordinatesQuick(self, authority, position, player):
        legal_move_coordinates = authority.LegalMoveCoordinates(position, player)
        move_to_choice_probability = {}
        other_player = authority.OtherPlayer(player)
        for move_coordinates in legal_move_coordinates:
            #print ("ConvPredictor.ChooseMoveCoordinates(): move_coordinates = {}".format(move_coordinates))
            resulting_position, winner = authority.MoveWithMoveArrayCoordinates(position, player, move_coordinates)
            if player == 'O':
                resulting_position = authority.SwapPositions(resulting_position)
            move_coordinates = tuple(move_coordinates)  # To make it hashable
            if winner == player:
                return move_coordinates
            elif winner == other_player:
                move_to_choice_probability[move_coordinates] = -1
            elif winner == 'draw':
                move_to_choice_probability[move_coordinates] = 0
            else:
                resulting_position_tsr = torch.tensor(resulting_position, dtype=torch.float).unsqueeze(0)
                # print ("ConvPredictor.ChooseMoveCoordinates(): resulting_position_tsr = {}".format(resulting_position_tsr))
                predictionTsr = self.forward(resulting_position_tsr)[1].squeeze(0)
                expected_value = predictionTsr[0].item() - predictionTsr[2].item()
                move_to_choice_probability[move_coordinates] = expected_value
            #print("ConvPredictor.ChooseMoveCoordinates(): expected_value = {}".format(move_to_choice_probability[move_coordinates]))

        softmax_temperature = self.simulation_softmax_temperature  # Quick decision

        if softmax_temperature <= 0:  # Hard max
            highest_expected_value = -2.0
            chosen_move_coordinates = []
            for move, expected_value in move_to_choice_probability.items():
                if expected_value > highest_expected_value:
                    highest_expected_value = expected_value
                    chosen_move_coordinates = [move]
                elif expected_value == highest_expected_value:
                    chosen_move_coordinates.append(move)
            return random.choice(chosen_move_coordinates)

        # print("ConvPredictor.ChooseMoveCoordinates(): Before normalization, move_to_choice_probability = \n{}".format(move_to_choice_probability))
        # Normalize
        sum = 0
        for move, expected_value in move_to_choice_probability.items():
            sum += math.exp(expected_value/softmax_temperature)

        for move, expected_value in move_to_choice_probability.items():
            move_to_choice_probability[move] = (math.exp(expected_value/softmax_temperature) )/sum

        # Draw a random number
        random_draw = random.random()
        running_sum = 0
        for move, probability in move_to_choice_probability.items():
            running_sum += probability
            if running_sum >= random_draw:
                return move
        raise RuntimeError("ConvPredictorDirect.ChooseMoveCoordinates(): Summed the probabilities without reaching the random number {}".format(random_draw))



    def SetSimulationSoftmaxTemperature(self, temperature):
        self.simulation_softmax_temperature = temperature