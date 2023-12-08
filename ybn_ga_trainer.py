from ybn_controller import YbnController
import numpy as np
import random
from kesslergame import Scenario, GraphicsType, TrainerEnvironment
from EasyGA import GA

def generate_gene():
    result = []
    for _ in range(5):
        result.append(random.random())
    
    result.sort()
    return result


def fitness(chromosome):
# This is the fitness function that will be used to evolve the models
    scenario = Scenario(name='Test Scenario',
                        num_asteroids=10,
                        ship_states=[
                            {
                                'position': (400, 400),
                                'angle': 90,
                                'lives': 3,
                                'team': 1,
                                "mines_remaining": 3
                            },
                        ],
                        map_size=(1000, 800),
                        time_limit=60,
                        ammo_limit_multiplier=0,
                        stop_if_no_ammo=False)

    # Define Game Settings
    game_settings = {
        'perf_tracker': True,
        'graphics_type': GraphicsType.Tkinter,
        'realtime_multiplier': 100000000,
        'graphics_obj': None,
        'frequency': 30
    }
    
    game = TrainerEnvironment(settings=game_settings)

    # calculate scores
    scores = []
    accuracies = []
    asteroids_hits = []
    for _ in range(10):
        chromosome_values = [
            chromosome[0].value,
            chromosome[1].value,
            chromosome[2].value
        ]
        score, _ = game.run(scenario= scenario, controllers=[YbnController(chromosome_values)])

        accuracy = 0
        asteroids_hit = 0
        for team in score.teams:
            accuracy = team.accuracy
            asteroids_hit = team.asteroids_hit
        
        # fitness is a weighted average of accuracy and number of asteroids hit
        accuracy_weight = 1
        asteroids_hit_weight = 10
        sum_of_weighted_terms = (accuracy_weight * accuracy) + (asteroids_hit_weight * asteroids_hit)
        sum_of_weights = accuracy_weight + asteroids_hit_weight
        weighted_score = sum_of_weighted_terms / sum_of_weights
        
        scores.append(weighted_score)
        accuracies.append(accuracy)
        asteroids_hits.append(asteroids_hit)

    average_accuracy = np.mean(accuracies)
    average_asteroids_hits = np.mean(asteroids_hit)
    fitness = np.mean(scores)
    print("\nAverages:")
    print(f"Accuracy: {average_accuracy}, Asteroid hits: {np.mean(average_asteroids_hits)}, Fitness: {fitness}")
    print("\n")
    
    return fitness

def main():
    ga = GA()
    ga.gene_impl = lambda: generate_gene()
    ga.chromosome_length = 3
    ga.population_size = 100
    ga.target_fitness_type = 'max'
    ga.generation_goal = 10
    ga.fitness_function_impl = fitness  
    ga.evolve() 
    ga.print_best_chromosome()


if __name__ == '__main__':
    main()