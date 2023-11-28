# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Yuyang Zeng, Benaka Achar, Nandini Parekh

from kesslergame import KesslerController
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np


class YbnController(KesslerController):

    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        pass


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """
        thrust = 0
        turn_rate = 0
        fire = False
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "YBN Controller"