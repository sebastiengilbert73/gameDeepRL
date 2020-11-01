import abc
import numpy as np

class Simulator(abc.ABC):
    def __init__(self):
        super().__init__()

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


    @abc.abstractmethod
    def ChooseAMove(self, authority, position, player):
        pass



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