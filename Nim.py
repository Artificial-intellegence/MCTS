########################################
# CS63: Artificial Intelligence, Lab 4
# Spring 2022, Swarthmore College
########################################
# NOTE: you should not need to modify this file.
########################################

from BoardGames import _base_game

class Nim(_base_game):
    def __init__(self, pieces=7):
        self.pieces = pieces
        self.turn = 1
        self._moves = None
        self._terminal = None
        self._winner = None

    def __repr__(self):
        return "Nim: %d Turn: %d" % (self.pieces, self.turn)
        
    def _print_char(self, i):
        return i
    
    def makeMove(self, move):
        if move > self.pieces:
            raise AttributeError("Nim game invalid move")
        new_game = Nim(self.pieces-move)
        new_game.turn = self.turn * -1
        return new_game

    @property
    def availableMoves(self):
        if self._moves is None:
            if self.pieces >= 3:
                self._moves = [3, 2, 1]
            elif self.pieces == 2:
                self._moves = [2,1]
            elif self.pieces == 1:
                self._moves = [1]
            else:
                self._moves = []
        return self._moves

    @property
    def isTerminal(self):
        if self._terminal is None:
            if self.pieces == 0:
                self._terminal = True
            else:
                self._terminal = False
        return self._terminal

    @property
    def winner(self):
        if not self.isTerminal:
            raise AttributeError("non-terminal states have no winner")
        if self._winner is None:
            self._winner = self.turn
        return self._winner
    
