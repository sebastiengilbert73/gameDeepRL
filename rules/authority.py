import abc

class GameAuthority(abc.ABC):
    """
    Abstract class that holds the rules of the game
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def Move(self, currentPositionArr, player, moveArr):
        pass # return (positionArr, winner)

    @abc.abstractmethod
    def MoveWithMoveArrayCoordinates(self, current_position, player, move_array_coordinates):
        pass

    @abc.abstractmethod
    def Winner(self, positionArr, lastPlayerWhoPlayed):
        pass

    @abc.abstractmethod
    def LegalMovesMask(self, positionArr, player):
        pass

    @abc.abstractmethod
    def LegalMoveCoordinates(self, positionArr, player): # List of 4-element lists
        pass

    @abc.abstractmethod
    def PositionArrayShape(self):
        pass

    @abc.abstractmethod
    def MoveArrayShape(self):
        pass

    @abc.abstractmethod
    def InitialPosition(self):
        pass

    @abc.abstractmethod
    def SwapPositions(self, positionArr):
        pass

    @abc.abstractmethod
    def PlayersList(self):
        pass

    """@abc.abstractmethod
    def MoveWithString(self, currentPositionArr, player, dropCoordinatesAsString):
        pass # return (positionArr, winner)
    """

    @abc.abstractmethod
    def Display(self, positionArr):
        pass

    @abc.abstractmethod
    def RaiseAnErrorIfNoLegalMove(self):
        pass

    def DisplayGame(self, positionsList):
        for position in positionsList:
            self.Display(position)
            print()

    def OtherPlayer(self, player):
        players = self.PlayersList()
        if not player in players:
            raise ValueError("GameAuthority.OtherPlayer(): '{}' is not part of the players list".format(player))
        if player == players[0]:
            return players[1]
        else:
            return players[0]