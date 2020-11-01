import rules.authority
import numpy as np
import ast


class Authority(rules.authority.GameAuthority):
    def __init__(self):
        super().__init__()
        self.playersList = ['X', 'O']
        self.positionArrayShape = (2, 1, 3, 3)
        self.moveArrayShape = (1, 1, 3, 3)
        self.playerToPlaneIndexDic = {'X': 0, 'O': 1}

    def Move(self, currentPositionArr, player, moveArr):
        if moveArr.shape != self.moveArrayShape:
            raise ValueError("Authority.Move(): moveArr.shape ({}) is not (1, 1, 3, 3)".format(moveArr.shape))
        dropCoordinates = self.DropCoordinates(moveArr)
        return self.MoveWithCoordinates(currentPositionArr, player, dropCoordinates)

    def Winner(self, positionArr, lastPlayerWhoPlayed):
        Xwins = self.ThereIs3InARow(self.playerToPlaneIndexDic['X'], positionArr)
        Owins = self.ThereIs3InARow(self.playerToPlaneIndexDic['O'], positionArr)
        if Xwins:
            return 'X'
        if Owins:
            return 'O'
        else:
            if np.count_nonzero(positionArr) == 9:  # All squares are occupied
                return 'draw'
            else:
                return None

    def LegalMovesMask(self, positionArr, player):
        if positionArr.shape != self.positionArrayShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionArr ({}) is not {}".format(
                positionArr.shape, self.positionArrayShape))
        legalMovesMask = np.ones(self.moveArrayShape, dtype=np.uint8) # Initialized with ones, i.e legal moves
        for row in range(3):
            for column in range(3):
                if positionArr[0, 0, row, column] != 0 or positionArr[1, 0, row, column] != 0:
                    legalMovesMask[0, 0, row, column] = 0
        return legalMovesMask

    def PositionArrayShape(self):
        return self.positionArrayShape

    def MoveArrayShape(self):
        return self.moveArrayShape

    def InitialPosition(self):
        initialPosition = np.zeros(self.positionArrayShape, dtype=np.uint8)
        return initialPosition

    def SwapPositions(self, positionArr):
        swappedPosition = np.copy(positionArr)
        playerXPlaneNdx = self.playerToPlaneIndexDic['X']
        playerOPlaneNdx = self.playerToPlaneIndexDic['O']
        swappedPosition[playerXPlaneNdx] = positionArr[playerOPlaneNdx]
        swappedPosition[playerOPlaneNdx] = positionArr[playerXPlaneNdx]
        return swappedPosition

    def PlayersList(self):
        return self.playersList

    def MoveWithString(self, currentPositionArr, player, dropCoordinatesAsString):
        dropCoordinatesTuple = ast.literal_eval(dropCoordinatesAsString)
        return self.MoveWithCoordinates(currentPositionArr, player, dropCoordinatesTuple)

    def Display(self, positionArr):
        if positionArr.shape != self.positionArrayShape:  # (C, D, H, W)
            raise ValueError("Authority.Display(): The shape of positionArr ({}) is not (2, 1, 3, 3)".format(
                positionArr.shape))
        for row in range(3):
            for column in range(3):
                # occupancy = None
                if positionArr[self.playerToPlaneIndexDic['X'], 0, row, column] == 1:
                    print(' X ', end='', flush=True)
                elif positionArr[self.playerToPlaneIndexDic['O'], 0, row, column] == 1:
                    print(' O ', end='', flush=True)
                else:
                    print('   ', end='', flush=True)
                if column != 2:
                    print('|', end='', flush=True)
                else:
                    print('')  # new line
            if row != 2:
                print('--- --- ---')

    def RaiseAnErrorIfNoLegalMove(self):
        return True

    def ThereIs3InARow(self, planeNdx, positionArr):
        if positionArr.shape != self.positionArrayShape: # (C, D, H, W)
            raise ValueError("Authority.ThereIs3InARow(): The shape of positionArr ({}) is not (2, 1, 3, 3)".format(positionArr.shape))
        # Horizontal lines
        for row in range(3):
            theRowIsFull = True
            for column in range(3):
                if positionArr[planeNdx, 0, row, column] != 1:
                    theRowIsFull = False
                    break
            if theRowIsFull:
                return True

        # Vertical lines
        for column in range(3):
            theColumnIsFull = True
            for row in range(3):
                if positionArr[planeNdx, 0, row, column] != 1:
                    theColumnIsFull = False
                    break
            if theColumnIsFull:
                return True

        # Diagonal \
        diagonalBackslashIsFull = True
        for index in range(3):
            if positionArr[planeNdx, 0, index, index] != 1:
                diagonalBackslashIsFull = False
                break
        if diagonalBackslashIsFull:
            return True

        # Diagonal /
        diagonalSlashIsFull = True
        for index in range(3):
            if positionArr[planeNdx, 0, index, 2 - index] != 1:
                diagonalSlashIsFull = False
                break
        if diagonalSlashIsFull:
            return True

        # Otherwise
        return False

    def MoveWithCoordinates(self, currentPositionArr, player, dropCoordinates):
        if currentPositionArr.shape != self.positionArrayShape: # (C, D, H, W)
            raise ValueError("Authority.MoveWithCoordinates(): The shape of currentPositionArr ({}) is not (2, 1, 3, 3)".format(currentPositionArr.shape))
        if player != 'X' and player != 'O':
            raise ValueError("Authority.MoveWithCoordinates(): The player must be 'X' or 'O', not '{}'".format(player))
        if len(dropCoordinates) != 2:
            raise ValueError("Authority.MoveWithCoordinates(): dropCoordinates ({}) is not a 2-tuple".format(dropCoordinates))
        if dropCoordinates[0] < 0 or dropCoordinates[0] > 2 or dropCoordinates[1] < 0 or dropCoordinates[1] > 2:
            raise ValueError("Authority.MoveWithCoordinates(): dropCoordinates entries ({}) are not in the range [0, 2]".format(dropCoordinates))
        if currentPositionArr[0, 0, dropCoordinates[0], dropCoordinates[1]] != 0 or \
                currentPositionArr[1, 0, dropCoordinates[0], dropCoordinates[1]] != 0:
            raise ValueError("Authority.MoveWithCoordinates(): Attempt to drop in an occupied square ({})".format(dropCoordinates))
        newPositionArr = currentPositionArr.copy()
        newPositionArr[self.playerToPlaneIndexDic[player], 0, dropCoordinates[0], dropCoordinates[1]] = 1
        winner = self.Winner(newPositionArr, player)
        return newPositionArr, winner

    def DropCoordinates(self, moveArr):
        numberOfOnes = 0
        dropCoordinates = None
        for row in range(3):
            for column in range(3):
                if moveArr[0, 0, row, column] == 1:
                    numberOfOnes += 1
                    dropCoordinates = (row, column)
        if numberOfOnes != 1:
            raise ValueError("Authority.DropCoordinates(): The number of ones in moveArr ({}) is not one".format(numberOfOnes))
        return dropCoordinates