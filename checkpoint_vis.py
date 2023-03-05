import os, sys, time
import gym
import gym_examples

import neat 
import visualize
import parallel

import numpy as np



env = gym.make('gym_examples/a1-v1', 
            #    render_mode="human"
               )
env.reset()



def eval_func(x):
    genomes = x[0]
    config = x[1]

    for genome_id, genome in genomes:
        genome.fitness = 0 # initialize fitnes 
        net = neat.nn.FeedForwardNetwork.create(genome, config)    
        default_action = np.array([ 0, -0.25622471, 0.0072058, 
                                    0, -0.25622471, 0.0072058, 
                                    0, -0.25622471, 0.0072058, 
                                    0, -0.25622471, 0.0072058])        
        new_action = default_action
        env.reset()

        for _ in range(30):
            env.step([0, -0.25622471, 0.0072058, 
                      0, -0.25622471, 0.0072058, 
                      0, -0.25622471, 0.0072058, 
                      0, -0.25622471, 0.0072058])

        for _ in range(170):
            observation, reward, terminated, truncated, info = env.step(new_action)
            genome.fitness += reward
            output = net.activate(observation)
            new_action = output 
            # print("new_action:", new_action)

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
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = 0
    default_action = np.array([ 0, -0.25622471, 0.0072058, 
                                0, -0.25622471, 0.0072058, 
                                0, -0.25622471, 0.0072058, 
                                0, -0.25622471, 0.0072058])    
    new_action = default_action
    env.reset()

   
    for _ in range(30):
        env.step([0, -0.25622471, 0.0072058, 
                    0, -0.25622471, 0.0072058, 
                    0, -0.25622471, 0.0072058, 
                    0, -0.25622471, 0.0072058])
        print("step:", _)
        time.sleep(0.5)

    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(new_action)
        genome.fitness += reward
        output = net.activate(observation)
        new_action = output 
        # print("new_action:", new_action)

        # time.sleep(1)

        if terminated or truncated:
            print(genome.fitness)
            env.reset()
            break

    env.close()



def run(config_file, checkpoint_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_file)

    p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 1)
    best_demo(winner, config)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Create visualization.
    visualize.draw_net(config, winner, view=True, show_disabled=False)
    visualize.plot_stats(p, ylog=False, view=True)
    visualize.plot_species(p, view=True)

if __name__ == '__main__':
    local_dir = os.path.dirname("__file__")
    checkpoint_path = os.path.join(local_dir, sys.argv[1])
    config_path = os.path.join(local_dir, sys.argv[2])
    run(config_path, checkpoint_path)
