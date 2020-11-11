import abc
import numpy as np
import random

class Simulator(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def ChooseMoveCoordinates(self, authority, position, player):
        pass

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
            move_coordinates = self.ChooseMoveCoordinates(authority, position, player)
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
        return legal_move_to_statistics_list

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
                move_coordinates = self.ChooseMoveCoordinates(authority, position, current_player)
            else:
                move_coordinates = other_player_simulator.ChooseMoveCoordinates(authority, position, current_player)
            position, winner = authority.MoveWithMoveArrayCoordinates(position, current_player, move_coordinates)
            number_of_moves += 1
            positionsList.append(position)
            current_player = authority.OtherPlayer(current_player)
        return positionsList, winner




class RandomSimulator(Simulator):
    def __init__(self):
        super().__init__()

    def ChooseMoveCoordinates(self, authority, position, player):
        legal_move_coordinates = authority.LegalMoveCoordinates(position, player)
        return random.choice(legal_move_coordinates)

