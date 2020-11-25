import abc
import numpy as np
import random

class Simulator(abc.ABC):
    def __init__(self, final_decision_softmax_temperature=0.0):
        super().__init__()
        self.final_decision_softmax_temperature = final_decision_softmax_temperature

    @abc.abstractmethod
    def ChooseMoveCoordinatesQuick(self, authority, position, player):
        pass

    def ChooseMoveCoordinatesMonteCarlo(self, authority, maximum_number_of_moves, number_of_simulations,
                                        starting_position, player):
        legal_move_to_statistics_list = self.LegalMoveStatistics(authority, maximum_number_of_moves,
                                                                 number_of_simulations, starting_position,
                                                                 player)
        if self.final_decision_softmax_temperature <= 0:  # Hard max
            highest_expected_value = -2.0
            chosen_move_coordinates = None
            for (move, stats) in legal_move_to_statistics_list:
                expected_value = (stats[0] - stats[2])/(stats[0] + stats[1] + stats[2])
                if expected_value > highest_expected_value:
                    highest_expected_value = expected_value
                    chosen_move_coordinates = move
            if chosen_move_coordinates is None:
                raise ValueError(
                    "Simulator.ChooseMoveCoordinatesMonteCarlo(): chosen_move_coordinates is None. legal_move_to_statistics_list = {}".format(
                        legal_move_to_statistics_list))
            return chosen_move_coordinates

        # Softmax
        move_to_expected_value_dic = {move: (stats[0] - stats[2])/(stats[0] + stats[1] + stats[2]) for (move, stats) in legal_move_to_statistics_list}
        # Normalize
        sum = 0
        #backup_sum = 0  # Sure to be > 0
        move_to_choice_probability = {}
        for move, expected_value in move_to_expected_value_dic.items():
            #sum += math.exp(expected_value / self.final_decision_softmax_temperature) - 1.0
            sum += math.exp(expected_value / self.final_decision_softmax_temperature)
        if sum > 0:
            for move, expected_value in move_to_expected_value_dic.items():
                move_to_choice_probability[move] = (math.exp(expected_value / self.final_decision_softmax_temperature) )/ sum
        else:
            raise ValueError(
                "Simulator.ChooseMoveCoordinatesMonteCarlo(): sum of exponentials ({}) is not > 0. move_to_expected_value_dic = {}".format(sum, move_to_expected_value_dic))
        # print("ConvPredictor.ChooseMoveCoordinates(): move_to_choice_probability = \n{}".format(move_to_choice_probability))

        # Draw a random number
        random_draw = random.random()
        running_sum = 0
        for move, probability in move_to_choice_probability.items():
            running_sum += probability
            if running_sum >= random_draw:
                return move
        raise RuntimeError(
            "Simulator.ChooseMoveCoordinatesMonteCarlo(): Summed the probabilities without reaching the random number {}. move_to_choice_probability = {}".format(
                random_draw, move_to_choice_probability))

    def SimulateGame(self, authority, maximum_number_of_moves, starting_position=None,
                     player=None):
        if starting_position is None:
            starting_position = authority.InitialPosition()
        players = authority.PlayersList()
        if player is None:
            player = players[0]

        winner = None
        number_of_moves = 0
        position = starting_position
        positionsList = [position]
        while winner is None and number_of_moves < maximum_number_of_moves:
            move_coordinates = self.ChooseMoveCoordinatesQuick(authority, position, player)
            position, winner = authority.MoveWithMoveArrayCoordinates(position, player, move_coordinates)
            number_of_moves += 1
            positionsList.append(position)
            player = authority.OtherPlayer(player)
        return positionsList, winner

    def LegalMoveStatistics(self, authority, maximum_number_of_moves, number_of_simulations,
                                 starting_position, player):
        legal_moves_coords = authority.LegalMoveCoordinates(starting_position, player)
        # len(legal_moves_coords) = number_of_legal_moves; Each coords in a 4-element list
        if len(legal_moves_coords) == 0:
            return []
        legal_move_to_statistics_list = []
        for moveNdx in range(len(legal_moves_coords)):
            move_coordinates = legal_moves_coords[moveNdx]
            number_of_wins = 0
            number_of_draws = 0
            number_of_losses = 0
            other_player = authority.OtherPlayer(player)
            position_after_candidate_move, winner = authority.MoveWithMoveArrayCoordinates(
                starting_position, player, move_coordinates)
            if winner == player:
                legal_move_to_statistics_list.append((move_coordinates, (number_of_simulations, 0, 0)))
            elif winner == other_player:
                legal_move_to_statistics_list.append((move_coordinates, (0, 0, number_of_simulations)))
            elif winner == 'draw':
                legal_move_to_statistics_list.append((move_coordinates, (0, number_of_simulations, 0)))
            else: # None: The game is not finished
                for simulationNdx in range(number_of_simulations):
                    positionsList, winner = self.SimulateGame(authority, maximum_number_of_moves,
                                                              position_after_candidate_move,
                                                              other_player)
                    if winner == player:
                        number_of_wins += 1
                    elif winner == other_player:
                        number_of_losses += 1
                    else:
                        number_of_draws += 1
                legal_move_to_statistics_list.append((move_coordinates, (number_of_wins, number_of_draws, number_of_losses)))
        # Normalize
        legal_move_to_normalized_stats_list = []
        for (move, stats) in legal_move_to_statistics_list:
            normalized_stats = (stats[0]/number_of_simulations, stats[1]/number_of_simulations, stats[2]/number_of_simulations)
            legal_move_to_normalized_stats_list.append((move, normalized_stats))
        return legal_move_to_normalized_stats_list

    def SimulateAsymmetricGame(self, authority, other_player_simulator,
                              maximum_number_of_moves, starting_position=None,
                              starting_player=None):
        if starting_position is None:
            starting_position = authority.InitialPosition()
        if starting_player is None:
            players = authority.PlayersList()
            starting_player = players[0]

        winner = None
        number_of_moves = 0
        position = starting_position
        current_player = starting_player
        positionsList = [position]
        while winner is None and number_of_moves < maximum_number_of_moves:
            move_coordinates = None
            if current_player == starting_player:
                move_coordinates = self.ChooseMoveCoordinatesQuick(authority, position, current_player)
            else:
                move_coordinates = other_player_simulator.ChooseMoveCoordinatesQuick(authority, position, current_player)
            position, winner = authority.MoveWithMoveArrayCoordinates(position, current_player, move_coordinates)
            number_of_moves += 1
            positionsList.append(position)
            current_player = authority.OtherPlayer(current_player)
        return positionsList, winner

    def SimulateAsymmetricGameMonteCarlo(self, authority, other_player_simulator,
                              maximum_number_of_moves, number_of_simulations,
                              starting_position=None, starting_player=None):
        if starting_position is None:
            starting_position = authority.InitialPosition()
        if starting_player is None:
            players = authority.PlayersList()
            starting_player = players[0]

        winner = None
        number_of_moves = 0
        position = starting_position
        current_player = starting_player
        positionsList = [position]
        while winner is None and number_of_moves < maximum_number_of_moves:
            move_coordinates = None
            if current_player == starting_player:
                move_coordinates = self.ChooseMoveCoordinatesMonteCarlo(authority, maximum_number_of_moves,
                                                              number_of_simulations,
                                                              position, current_player)
            else:
                move_coordinates = other_player_simulator.ChooseMoveCoordinatesMonteCarlo(authority,
                                                                                maximum_number_of_moves,
                                                                                number_of_simulations,
                                                                                position, current_player)
            position, winner = authority.MoveWithMoveArrayCoordinates(position, current_player, move_coordinates)
            number_of_moves += 1
            positionsList.append(position)
            current_player = authority.OtherPlayer(current_player)
        return positionsList, winner

    """
    def ChooseAMoveMonteCarlo(self, authority, maximum_number_of_moves, number_of_simulations,
                                 starting_position, player):
        legal_move_to_statistics_list = self.LegalMoveStatistics(authority, maximum_number_of_moves,
                                                                 number_of_simulations, starting_position,
                                                                 player)
        highest_expected_value = -2.0
        chosen_move_coordinates = None
        for (move, statistics) in legal_move_to_statistics_list:
            expected_value = (statistics[0] - statistics[2])/number_of_simulations  # win_rate - loss_rate
            #print("ChooseAMoveMonteCarlo(): move = {}; statistics = {}".format(move, statistics))
            if expected_value > highest_expected_value:
                highest_expected_value = expected_value
                chosen_move_coordinates = move
        if chosen_move_coordinates is None:
            raise ValueError("Authority.ChooseAMoveMonteCarlo(): chosen_move_coordinates is None. legal_move_to_statistics_list = {}".format(legal_move_to_statistics_list))
        return chosen_move_coordinates
    """
    def SetFinalDecisionSoftmaxTemperature(self, temperature):
        self.final_decision_softmax_temperature = temperature


class RandomSimulator(Simulator):
    def __init__(self):
        super().__init__()

    def ChooseMoveCoordinatesQuick(self, authority, position, player):
        legal_move_coordinates = authority.LegalMoveCoordinates(position, player)
        return random.choice(legal_move_coordinates)

