# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Yuyang Zeng, Benaka Achar, Nandini Parekh

from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
from EasyGA import GA
import random
from kesslergame import Scenario, GraphicsType, TrainerEnvironment, KesslerGame, KesslerController
import sys


class YbnController(KesslerController):
    def build_2_custom_trimfs(universe, gene):
        # membership function must cover the min and max of the universe
        min = universe[0]
        max = universe[-1]

        gene.sort()
        
        # map random normalized value to the range of the universe
        m1 = max * gene[0]
        m2 = max * gene[1]

        first_trimf = fuzz.trimf(universe, [min, m1, m2])
        second_trimf = fuzz.trimf(universe, [m1, m2, max])

        return first_trimf, second_trimf

    def build_5_custom_trimfs(universe, gene):
        # membership function must cover the min and max of the universe
        min = universe[0]
        max = universe[-1]
        
        # map random normalized value to the range of the universe
        m2 = max * gene[0]
        m1 = max * gene[1]
        
        if min < 0:
            m1 = -1 * gene[0]
            m2 = -1 * gene[1]

        m3 = max * gene[2]
        m4 = max * gene[3]
        m5 = max * gene[4]

        # gene is list of 5 numbers between 0 and 1
        # chromosome has 3 genes

        # ship_run['NL']  = fuzz.trimf(ship_run.universe, [-150,-150,-30])
        # ship_run['NS']  = fuzz.trimf(ship_run.universe, [-150,-80,0])
        # ship_run['M']  = fuzz.trimf(ship_run.universe, [-50,0,50])
        # ship_run['PS']  = fuzz.trimf(ship_run.universe, [0,50,90])
        # ship_run['PL'] = fuzz.trimf(ship_run.universe, [0,150,150])

        # hit_dist['NL']  = fuzz.trimf(hit_dist.universe, [0,0,200])
        # hit_dist['NS']  = fuzz.trimf(hit_dist.universe, [100,300,600])
        # hit_dist['M']  = fuzz.trimf(hit_dist.universe, [200,400,600])
        # hit_dist['PS']  = fuzz.trimf(hit_dist.universe, [200,600,800])
        # hit_dist['PL'] = fuzz.trimf(hit_dist.universe, [400,800,800])
        first_trimf = fuzz.trimf(universe, [min, min, m3])
        second_trimf = fuzz.trimf(universe, [m2, m1, m3])
        third_trimf = fuzz.trimf(universe, [m1, m3, m4])
        fourth_trimf = fuzz.trimf(universe, [m3, m4, m5])
        fifth_trimf = fuzz.trimf(universe, [m3, max, max])
        return (first_trimf, second_trimf, third_trimf, fourth_trimf, fifth_trimf)

    def __init__(self, chromosome):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.eval_frames = 0  # Initialize the frame as the start
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python

        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')

        ship_run = ctrl.Consequent(np.arange(-150,150,10), 'ship_run')
        hit_dist = ctrl.Antecedent(np.arange(0,800,10), 'hit_dist')
        ast_size = ctrl.Antecedent(np.arange(0,5,0.01), 'ast_size')

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi/6,math.pi/3)

        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-30])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90,-30,0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-30,0,30])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,30,90])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30,180,180])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        #Declare fuzzy sets for the ship_run consequent; this will be returned as thrust.
        nl_trimf, ns_trimf, m_trimf, ps_trimf, pl_trimf = YbnController.build_5_custom_trimfs(ship_run.universe, chromosome[0].value)
        ship_run['NL']  = nl_trimf
        ship_run['NS']  = ns_trimf
        ship_run['M']  = m_trimf
        ship_run['PS']  = ps_trimf
        ship_run['PL'] = pl_trimf

        nl_trimf, ns_trimf, m_trimf, ps_trimf, pl_trimf = YbnController.build_5_custom_trimfs(hit_dist.universe, chromosome[1].value)
        hit_dist['NL']  = nl_trimf
        hit_dist['NS']  = ns_trimf
        hit_dist['M']  = m_trimf
        hit_dist['PS']  = ps_trimf
        hit_dist['PL'] = pl_trimf

        small_trimf, large_trimf = YbnController.build_2_custom_trimfs(ast_size.universe, chromosome[2].value)
        ast_size['S'] = small_trimf
        ast_size['L'] = large_trimf

        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        rule16 = ctrl.Rule(hit_dist['NL'] & ast_size['S'] , (ship_run['NS']))
        rule17 = ctrl.Rule(hit_dist['NS'] & ast_size['S'], (ship_run['NS']))
        rule18 = ctrl.Rule(hit_dist['M'] & ast_size['S'], (ship_run['M']))
        rule19 = ctrl.Rule(hit_dist['PS'] & ast_size['S'], (ship_run['PS']))
        rule20 = ctrl.Rule(hit_dist['PL'] & ast_size['S'], (ship_run['PL']))
        
        rule21 = ctrl.Rule(hit_dist['NL'] & ast_size['L'], (ship_run['NL']))
        rule22 = ctrl.Rule(hit_dist['NS'] & ast_size['L'], (ship_run['NS']))
        rule23 = ctrl.Rule(hit_dist['M'] & ast_size['L'], (ship_run['M']))
        rule24 = ctrl.Rule(hit_dist['PS'] & ast_size['L'], (ship_run['PS']))
        rule25 = ctrl.Rule(hit_dist['PL'] & ast_size['L'], (ship_run['PL']))

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)
        self.targeting_control.addrule(rule22)
        self.targeting_control.addrule(rule23)
        self.targeting_control.addrule(rule24)
        self.targeting_control.addrule(rule25)

        
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        # Get position of the ship 
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)

            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist


        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # ship_speed = math.sqrt(((ship_state["velocity"][0]))**2 + ((ship_state["velocity"][1]))**2)
        # if ship_speed != 0:
        #     hit_time = abs( closest_asteroid["dist"] / ship_speed)
        # else:
        #     hit_time = 200  # maximum time


        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['hit_dist'] = closest_asteroid["dist"]
        shooting.input['ast_size'] = closest_asteroid["aster"]["size"]

        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
       
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        # print(shooting.output['ship_run'])
        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = shooting.output['ship_run']
        dis = closest_asteroid["dist"]
        drop_mine = False


        self.eval_frames +=1   # do increment for each frame, since we should call action at every each frame of the game.
        return thrust, turn_rate, fire, drop_mine

  
    
    @property
    def name(self) -> str:
        return "YBN Controller"


def generate_gene():
    result = []
    for _ in range(5):
        result.append(random.random())
    
    result.sort()
    return result


def fitness(chromosome, train = True):
# This is the fitness function that will be used to evolve the models
    my_test_scenario = Scenario(name='Test Scenario',
                        num_asteroids=10,
                        ship_states=[
                            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},],
                        map_size=(1000, 800),
                        time_limit=60,
                        ammo_limit_multiplier=0,
                        stop_if_no_ammo=False)

    # Define Game Settings
    game_settings = {'perf_tracker': True,
                'graphics_type': GraphicsType.Tkinter,
                'realtime_multiplier': 1,
                'graphics_obj': None,
                'frequency': 30}
    
    game = None 
    if train:
        game = TrainerEnvironment(settings=game_settings)
    else:
        game = KesslerGame(settings=game_settings)

    # calculate scores
    scores = []
    for _ in range(10):
        score, perf_data = game.run(scenario= my_test_scenario, controllers= [YbnController(chromosome)])  # Use this to visualize the game scenario)

        accuracy = 0
        asteroids_hit = 0
        for team in score.teams:
            accuracy = team.accuracy
            asteroids_hit = team.asteroids_hit
            
        accuracy_weight = 1
        asteroids_hit_weight = 10

        weighted_score = ((accuracy_weight * accuracy) + (asteroids_hit_weight * asteroids_hit)) / (accuracy_weight + asteroids_hit_weight)
        print(f"Accuracy: {accuracy}, Asteroids hit: {asteroids_hit}, Weighted Score: {weighted_score}")
        scores.append(weighted_score)

    return np.mean(scores)

def main():
    if (len(sys.argv) > 1):
        ga = GA()
        ga.gene_impl = lambda: generate_gene()
        ga.chromosome_length = 3
        ga.population_size = 200
        ga.target_fitness_type = 'max'
        ga.generation_goal = 20
        ga.fitness_function_impl = fitness  
        ga.evolve() 
        ga.print_best_chromosome()
    else:
        ga = GA()
        best_chromosome = ga.make_chromosome([
            [0.18359739372655026, 0.24546060845320483, 0.32585904178523684, 0.4017497974244143, 0.881590845180744],
            [0.1460242522377897, 0.2216150705969483, 0.5117562727550719, 0.7263147713160476, 0.7363169119439348],
            [0.14602900868249313, 0.24390071596487917, 0.3369521757342465, 0.6312268621285418, 0.9613523653547065]
        ])

        
        f = fitness(best_chromosome, train=False)
        print(f)

        

if __name__ == '__main__':
    main()