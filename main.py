import os
import gym
import gym_examples

import neat 
import visualize
import parallel

import numpy as np

# Hyper-parameters
EVOLUTION_NUMBER = 20



env = gym.make('gym_examples/a1-v1', 
               render_mode="human"
               )
env.reset()



def eval_func(x):
    genomes = x[0]
    config = x[1]

    for genome_id, genome in genomes:
        genome.fitness = 0 # initialize fitnes 
        net = neat.nn.FeedForwardNetwork.create(genome, config)    
        default_action = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.,9 -1.8])
        new_action = default_action
        env.reset()

        for _ in range(500):
            observation, reward, terminated, truncated, info = env.step(new_action)
            genome.fitness += reward
            output = net.activate(observation)
            new_action = output

            if terminated or truncated:
                env.reset()
                break
    
    env.close()



def eval_genomes(genomes, config):
    paraEval = parallel.ParallelEvaluator(8, eval_func)
    paraEval.eval_function((genomes, config))



def best_demo(genome, config):
    '''
        demonstrate best performing genome.
    '''
    env = gym.make('gym_examples/a1-v1', render_mode="human")
    env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    default_action = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.,9 -1.8])
    new_action = default_action
   
    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(new_action)
        output = net.activate(observation) 
        new_action = output
        genome.fitness += reward
        print("step", _, " fitness:" , genome.fitness)
   
    env.close()



def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, EVOLUTION_NUMBER)

    best_demo(winner, config)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Create visualization.
    # node_names = {-1: 'Cart_Pos', 
    #               -2: 'Cart_Vel', 
    #               -3:'Pole_Ang' , 
    #               -4:'Pole_Ang_vel',
    #               0:'Action'
    #                }
    visualize.draw_net(config, 
                       winner, 
                       view=True, 
                    #    node_names=node_names,
                       show_disabled=False,
                       )
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, 'config-neat.txt')
    observation, info = env.reset()    
    run(config_path)
    env.close()