import rules.authority
import numpy as np
import ast

class Authority(rules.authority.GameAuthority):
    # Must implement:
    # Move(self, currentPositionArr, player, moveArr)   y
    # MoveWithMoveArrayCoordinates(self, current_position, player, move_array_coordinates)  y
    # Winner(self, positionArr, lastPlayerWhoPlayed)    y
    # LegalMovesMask(self, positionArr, player)     y
    # LegalMoveCoordinates(self, positionArr, player)  # List of 4-element lists    y
    # PositionArrayShape(self)  y
    # MoveArrayShape(self)  y
    # InitialPosition(self)     y
    # SwapPositions(self, positionArr)  y
    # PlayersList(self)     y
    # Display(self, positionArr)    y
    # RaiseAnErrorIfNoLegalMove(self) y
    #
    def __init__(self, numberOfRows=6, numberOfColumns=7):
        super().__init__()
        if numberOfColumns < 4 or numberOfRows < 4:
            raise ValueError("Authority.__init__(): The number of rows ({}) and the number of columns ({}) must be at least 4".format(numberOfRows, numberOfColumns))
        self.playersList = ['yellow', 'red']
        self.positionArrayShape = (2, 1, numberOfRows, numberOfColumns)
        self.moveArrayShape = (1, 1, 1, numberOfColumns)
        self.playerToPlaneIndexDic = {'yellow': 0, 'red': 1}
        self.numberOfRows = numberOfRows
        self.numberOfColumns = numberOfColumns

    def PlayersList(self):
        return self.playersList

    def ThereIs4InARow(self, planeNdx, positionArray):
        if positionArray.shape != self.positionArrayShape: # (C, D, H, W)
            raise ValueError("Authority.ThereIs4InARow(): The shape of positionArray ({}) is not {}".format(positionArray.shape, self.positionArrayShape))

        for row in range(self.numberOfRows):
            for column in range(self.numberOfColumns):
                # Left end of a horizontal line
                if column < self.numberOfColumns - 3:
                    thereIsAHorizontalLine = True
                    deltaColumn = 0
                    while deltaColumn < 4 and thereIsAHorizontalLine:
                        if positionArray[planeNdx, 0, row , column + deltaColumn] != 1:
                            thereIsAHorizontalLine = False
                        deltaColumn += 1
                    if thereIsAHorizontalLine:
                        return True

                # Upper end of a vertical line
                if row < self.numberOfRows - 3:
                    thereIsAVerticalLine = True
                    deltaRow = 0
                    while deltaRow < 4 and thereIsAVerticalLine:
                        if positionArray[planeNdx, 0, row + deltaRow, column] != 1:
                            thereIsAVerticalLine = False
                        deltaRow += 1
                    if thereIsAVerticalLine:
                        return True

                # North-West end of a \
                if row < self.numberOfRows - 3 and column < self.numberOfColumns - 3:
                    thereIsABackSlash = True
                    deltaRowColumn = 0
                    while deltaRowColumn < 4 and thereIsABackSlash:
                        if positionArray[planeNdx, 0, row + deltaRowColumn, column + deltaRowColumn] != 1:
                            thereIsABackSlash = False
                        deltaRowColumn += 1
                    if thereIsABackSlash:
                        return True

                # North-East end of a /
                if row < self.numberOfRows - 3 and column >= 3:
                    thereIsASlash = True
                    deltaRowColumn = 0
                    while deltaRowColumn < 4 and thereIsASlash:
                        if positionArray[planeNdx, 0, row + deltaRowColumn, column - deltaRowColumn] != 1:
                            thereIsASlash = False
                        deltaRowColumn += 1
                    if thereIsASlash:
                        return True
        # Otherwise
        return  False

    def MoveWithColumn(self, currentPositionArr, player, dropColumn):
        if currentPositionArr.shape != self.positionArrayShape: # (C, D, H, W)
            raise ValueError("Authority.MoveWithColumn(): The shape of currentPositionArr {} is not {}".format(currentPositionArr.shape, self.positionArrayShape))
        if dropColumn >= self.numberOfColumns:
            raise ValueError("Authority.MoveWithColumn(): dropColumn ({}) is >= self.numberOfColumns ({})".format(dropColumn, self.numberOfColumns))
        topAvailableRow = self.TopAvailableRow(currentPositionArr, dropColumn)
        if topAvailableRow == None:
            raise ValueError(
                "Authority.MoveWithColumn(): Attempt to drop in column {}, while it is already filled".format(
                    dropColumn))
        newPositionArr = np.copy(currentPositionArr)
        newPositionArr[self.playerToPlaneIndexDic[player], 0, topAvailableRow, dropColumn] = 1.0
        winner = self.Winner(newPositionArr, player)
        return newPositionArr, winner

    def TopAvailableRow(self, positionArr, dropColumn):
        # Must return None if the column is already filled
        # Check the bottom row
        if positionArr[0, 0, self.numberOfRows - 1, dropColumn] == 0 and \
            positionArr[1, 0, self.numberOfRows - 1, dropColumn] == 0:
            return self.numberOfRows - 1

        highestOneRow = self.numberOfRows - 1
        for row in range(self.numberOfRows - 2, -1, -1): # Count backward: 4, 3, 2, 1, 0
            if positionArr[0, 0, row, dropColumn] > 0 or \
                positionArr[1, 0, row, dropColumn] > 0:
                highestOneRow = row
        if highestOneRow == 0: # The column is already filled
            return None
        else:
            return highestOneRow - 1

    def Move(self, currentPositionArr, player, moveArr):
        if moveArr.shape != self.moveArrayShape:
            raise ValueError("Authority.Move(): moveArr.shape ({}) is not {}".format(moveArr.shape, self.moveArrayShape))
        numberOfOnes = 0
        dropColumn = None

        for column in range(self.numberOfColumns):
            if moveArr[0, 0, 0, column] == 1:
                numberOfOnes += 1
                dropColumn = column
        if numberOfOnes != 1:
            raise ValueError("Authority.Move(): The number of ones in moveArr ({}) is not one".format(numberOfOnes))
        return self.MoveWithColumn(currentPositionArr, player, dropColumn)

    def InitialPosition(self):
        initialPosition = np.zeros(self.positionArrayShape)
        return initialPosition

    def MoveArrayShape(self):
        return self.moveArrayShape

    def PositionArrayShape(self):
        return self.positionArrayShape

    def Winner(self, positionArray, lastPlayerWhoPlayed):
        lastPlayerPlane = self.playerToPlaneIndexDic[lastPlayerWhoPlayed]
        if self.ThereIs4InARow(lastPlayerPlane, positionArray):
            return lastPlayerWhoPlayed
        else:
            if np.count_nonzero(positionArray) == self.numberOfRows * self.numberOfColumns: # All spots are occupied
                return 'draw'
            else:
                return None

    def LegalMovesMask(self, positionArray, player):
        if positionArray.shape != self.positionArrayShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionArray ({}) is not {}".format(
                positionArray.shape, self.positionArrayShape))
        legalMovesMask = np.ones(self.moveArrayShape, dtype=np.int8)  # Initialized with ones, i.e legal moves
        for column in range(self.numberOfColumns):
            if positionArray[0, 0, 0, column] != 0 or positionArray[1, 0, 0, column] != 0:
                legalMovesMask[0, 0, 0, column] = 0
        return legalMovesMask

    def SwapPositions(self, positionArr):
        swappedPosition = np.zeros(self.positionArrayShape)
        swappedPosition[[0, 1], :] = positionArr[[1, 0], :]  # Cf. https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-150.php
        return swappedPosition

    def Display(self, positionArr):
        planeNdxToSymbolDic = {0: 'y', 1: 'r'}
        for row in range(self.numberOfRows):
            for column in range(self.numberOfColumns):
                if positionArr[0, 0, row, column] > 0:
                    print ('{} '.format(planeNdxToSymbolDic[0]), end='')
                elif positionArr[1, 0, row, column] > 0:
                    print ('{} '.format(planeNdxToSymbolDic[1]), end='')
                else:
                    print ('. ', end='')
            print('\n')
        print("**************")

    def RaiseAnErrorIfNoLegalMove(self):
        return True

    def MoveWithMoveArrayCoordinates(self, current_position, player, move_array_coordinates):
        dropColumn = move_array_coordinates[3]
        return self.MoveWithColumn(current_position, player, dropColumn)

    def LegalMoveCoordinates(self, positionArr, player):  # List of 4-element lists
        legal_move_coordinates = []
        for columnNdx in range(self.numberOfColumns):
            if positionArr[0, 0, 0, columnNdx] == 0 and positionArr[1, 0, 0, columnNdx] == 0:
                legal_move_coordinates.append([0, 0, 0, columnNdx])
        return legal_move_coordinates