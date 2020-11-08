import logging
import simulation.simulator
import rules.tictactoe
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')


def main():
    logging.info("test.py main()")

    simulator = simulation.simulator.RandomSimulator()
    authority = rules.tictactoe.Authority()
    maximum_number_of_moves = 9
    number_of_simulations = 1000
    starting_position = authority.InitialPosition()

    starting_position[0, 0, 0, 0] = 1
    starting_position[0, 0, 0, 1] = 1
    starting_position[0, 0, 1, 0] = 1
    starting_position[1, 0, 1, 2] = 1
    starting_position[1, 0, 2, 1] = 1
    starting_position[1, 0, 2, 2] = 1


    authority.Display(starting_position)

    logging.info("main(): Before simulator.LegalMoveStatistics()")
    legal_move_to_stats_list = simulator.LegalMoveStatistics(authority,
                                                             maximum_number_of_moves,
                                                             number_of_simulations,
                                                             starting_position,
                                                             'X')
    logging.info("main(): After simulator.LegalMoveStatistics()")
    for (move, stats) in legal_move_to_stats_list:
        print ("{}: {}".format(move, stats))





if __name__ == '__main__':
    main()