import os, sys
import gym
import gym_examples

import neat 
import visualize

import numpy as np



env = gym.make('gym_examples/a1-v1', 
            #    render_mode="human"
               )
env.reset()



def eval_func(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0 # initialize fitnes 
        net = neat.nn.FeedForwardNetwork.create(genome, config)    
        default_action = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.,9 -1.8])
        new_action = default_action
        env.reset()

        for _ in range(100):
            observation, reward, terminated, truncated, info = env.step(new_action)
            genome.fitness += reward
            output = net.activate(observation)
            new_action = output

            if terminated or truncated:
                env.reset()
                break
    
    env.close()



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
        print("step", _, " fitness:" , genome.fitness, "Terminated:", terminated)
   
    env.close()



def run(config_file, checkpoint_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    print(checkpoint_file)
    p.run(eval_func, 1)
    best_demo(p.best_genome, config)

    # show final stats
    print('\nBest genome:\n{!s}'.format(p))

    # Create visualization.
    # node_names = {-1: 'Cart_Pos', 
    #               -2: 'Cart_Vel', 
    #               -3:'Pole_Ang' , 
    #               -4:'Pole_Ang_vel',
    #               0:'Action'
    #                }
    visualize.draw_net(config, 
                       p, 
                       view=True, 
                    #    node_names=node_names,
                       show_disabled=False,
                       )
    visualize.plot_stats(p, ylog=False, view=True)
    visualize.plot_species(p, view=True)

if __name__ == '__main__':

    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, 'config-neat.txt')
    checkpoint_path = os.path.join(local_dir, sys.argv[1])

    run(config_path, checkpoint_path)
