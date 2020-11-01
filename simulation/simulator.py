import abc
import numpy as np

class Simulator(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def ChooseAMove(self, authority, position, player):
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
            moveArr = self.ChooseAMove(authority, position, player)
            position, winner = authority.Move(position, player, moveArr)
            number_of_moves += 1
            positionsList.append(position)
            player = authority.OtherPlayer(player)
        return positionsList, winner

    def LegalMoveStatistics(self, authority, maximum_number_of_moves, number_of_simulations,
                                 starting_position, player):
        legal_moves_mask = authority.LegalMovesMask(starting_position, player)
        legal_moves_coords = np.transpose(np.nonzero(legal_moves_mask))
        # legal_moves_coords.shape = (number_of_legal_moves, 4)
        if legal_moves_coords.shape[0] == 0:
            return []
        legal_move_to_statistics_list = []
        for moveNdx in range(legal_moves_coords.shape[0]):
            candidate_move_arr = np.zeros(authority.MoveArrayShape(), dtype=np.uint8)
            candidate_move_arr[legal_moves_coords[moveNdx, 0], legal_moves_coords[moveNdx, 1],
                        legal_moves_coords[moveNdx, 2], legal_moves_coords[moveNdx, 3]] = 1
            number_of_wins = 0
            number_of_draws = 0
            number_of_losses = 0
            other_player = authority.OtherPlayer(player)
            position_after_candidate_move, winner = authority.Move(starting_position, player, candidate_move_arr)
            if winner == player:
                legal_move_to_statistics_list.append((candidate_move_arr, (number_of_simulations, 0, 0)))
            elif winner == other_player:
                legal_move_to_statistics_list.append((candidate_move_arr, (0, 0, number_of_simulations)))
            elif winner == 'draw':
                legal_move_to_statistics_list.append((candidate_move_arr, (0, number_of_simulations, 0)))
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
                legal_move_to_statistics_list.append((candidate_move_arr, (number_of_wins, number_of_draws, number_of_losses)))
        return legal_move_to_statistics_list


class RandomSimulator(Simulator):
    def __init__(self):
        super().__init__()

    def ChooseAMove(self, authority, position, player):
        legal_moves_mask = authority.LegalMovesMask(position, player)
        legal_moves_coords = np.transpose(np.nonzero(legal_moves_mask))
        # legal_moves_coords.shape = (number_of_legal_moves, 4)
        if legal_moves_coords.shape[0] == 0:
            return None
        chosen_move_ndx = np.random.randint(legal_moves_coords.shape[0])
        chosen_move = np.zeros(authority.MoveArrayShape(), dtype=np.uint8)
        chosen_move[legal_moves_coords[chosen_move_ndx, 0], legal_moves_coords[chosen_move_ndx, 1],
            legal_moves_coords[chosen_move_ndx, 2], legal_moves_coords[chosen_move_ndx, 3]] = 1
        return chosen_move
