import unittest
from sea_battle import SeaBattleBoard

def has_right_ship_count(board: SeaBattleBoard) -> bool:
    squares_occupied = set()
    ships = [1, 2, 3, 4]
    for ship in board.ships:
        size = 0
        for square in ship["location"]:
            size += 1
            if square in squares_occupied:
                return False
            else:
                squares_occupied.add(square)
        try:
            ships.remove(size)
        except ValueError:
            return False
    if len(ships) != 0:
        return False
    return True

def get_adjacent_squares(square):
    row, col = square
    adjacent_squares = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 10:
                adjacent_squares.append((new_row, new_col))

    return adjacent_squares


def no_adjacent_ships(board: SeaBattleBoard) -> bool:
    for ship in board.ships:
        for square in ship["location"]:
            for adjacent_square in get_adjacent_squares(square):
                # Check if the adjacent square belongs to another ship
                for other_ship in board.ships:
                    if other_ship != ship and adjacent_square in other_ship["location"]:
                        return False
    return True


    
def is_valid_board(board):
    return has_right_ship_count(board) and no_adjacent_ships(board)


class TestSeaBattleBoard(unittest.TestCase):
    def test_place_ships_randomly(self):
        for _ in range(100):  # Run the test for 100 iterations
            board = SeaBattleBoard()
            board.place_ships_randomly()
            if not is_valid_board(board):
                print("Invalid board:")
                print(board.__str__)
                self.assertTrue(is_valid_board(board))
            else:
                self.assertTrue(is_valid_board(board))


if __name__ == '__main__':
    unittest.main()
