import gym
import neat
import os
import visualize
import parallel

import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
import datetime



#========== PARAMS ========================================================
NUM_EVOLUTIONS = 200
NUM_EVAL_STEPS = 500
NUM_EXPERIMENTS = 1
NUM_INPUTS = 4
NUM_OUTPUT = 1
FITNESS_FUNC_ALPHA = 1
NUM_CORES = 8
#========== ENVIRONMENT ===================================================
#   ACTION SPACE: (1,) values {0, 1}
#       - 0: Push cart to the left
#       - 1: Push cart to the right
#        
#   OBSERVATION SPACE:
#       - 0: Cart position {-4.8 ~ 4.8}
#       - 1: Cart Velocity {-Inf ~ Inf}
#       - 2: Pole Angle {-0.418 rad (-24 deg) ~ 0.418 rad (24 deg)}
#       - 3: Pole Angular Velocity {-Inf ~ Inf}
#
#   REWARDS:
#       A reward of +1 for every step taken 
#===========================================================================



# Set environment.
env = gym.make("CartPole-v1",
            # render_mode="human"
            )



def fitness_func(observation, reward, alpha):
    '''
        Fitness function calculates the final reward value.
    '''
    pos = True
    if pos:
        cart_pos = abs(observation[0]) * alpha
        return reward - cart_pos
    else: 
        return reward



def eval_func(x):
    ''' Evaluation function of OpenAI gym Cartpole-v1

        ACTION SPACE: (1,) values {0, 1}
            - 0: Push cart to the left
            - 1: Push cart to the right
        
        OBSERVATION SPACE:
            - 0: Cart position {-4.8 ~ 4.8}
            - 1: Cart Velocity {-Inf ~ Inf}
            - 2: Pole Angle {-0.418 rad (-24 deg) ~ 0.418 rad (24 deg)}
            - 3: Pole Angular Velocity {-Inf ~ Inf}

        REWARDS:
            A reward of +1 for every step taken 
    '''
    # decompose x to genomes and config
    genomes = x[0]
    config = x[1]

    for genome_id, genome in genomes:
        genome.fitness = 0  # initialize fitness 
        net = neat.nn.FeedForwardNetwork.create(genome, config) 
        new_action = env.action_space.sample() # initially take random input
        env.reset()

        for _ in range(NUM_EVAL_STEPS):
            observation, reward, terminated, truncated, info = env.step(new_action)
            output = net.activate(observation) # between -1.0 ~ 1.0
            output = 1 if output[0] > 0 else 0 # set output either 0 or 1
            new_action = output
            genome.fitness += fitness_func(observation, reward, FITNESS_FUNC_ALPHA)

            if terminated or truncated:
                env.reset()
                break



def data_plot(data, labels):
    '''
        Plot observation data and corresponding lables
    '''
    data = np.delete(data, 0, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot()
    x = data[0]
    ax.clear()
    ax.set_xlabel('episode')   

    for d, l in zip(data[1:], labels):
        ax.plot(x, d, label=l)
    
    ax.legend()
    plt.show()



def best_demo(genome, config, render_plot=False):
    '''
        demonstrate best performing genome.
    '''
    env = gym.make("CartPole-v1")
    observation, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    new_action = env.action_space.sample()
    observation_data = np.zeros((len(observation)+2, 1)) # Store observation data

    for _ in range(NUM_EVAL_STEPS):
        observation, reward, terminated, truncated, info = env.step(new_action)
        output = net.activate(observation) # between -1.0 ~ 1.0
        output_abs = 1 if output[0] > 0 else 0 # set output either 0 or 1
        new_action = output_abs
        genome.fitness += fitness_func(observation, reward, FITNESS_FUNC_ALPHA)

        if terminated or truncated:
            env.reset()
            break

        if render_plot:
            # Get for plotting data 
            obs = observation
            obs = np.insert(obs, 0, output, axis=0)
            obs = np.insert(obs, 0, _, axis=0)
            obs = obs.reshape(len(obs), 1)
            observation_data = np.append(observation_data, obs, axis=1)

    env.close()  

    if render_plot:
        data_labels=["activation", 
                     "cart_pos", 
                     "cart_vel", 
                     "pole_angle", 
                     "pole_angle_velocity"]
        data_plot(observation_data, data_labels)



def parallel_eval_genomes(genomes, config):
    paraEval = parallel.ParallelEvaluator(NUM_CORES, eval_func)
    paraEval.eval_function((genomes, config))



def run_stats(stats):
    data = {}
    data['generations'] = len(stats.get_fitness_mean())
    data['num_n'] = len(list(stats.best_genome().nodes))
    data['num_c'] = len(list(stats.best_genome().connections))
    data['bias'] = list(stats.best_genome().nodes.values())[0].bias
    data['weights'] = dict((k, []) for k in range(-NUM_INPUTS, 0))

    for c in stats.best_genome().connections.values():
        input_node = c.key[0]
        if input_node < 0:
            data['weights'][input_node].append(c.weight)
    print("++++++++++++++++", data)
    
    return data
    


def run(config_file, view_stats=False):
    '''
        runs the NEAT algorithm to train a neural network
    '''
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)     
    p.add_reporter(neat.StdOutReporter(True))   # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    winner = p.run(parallel_eval_genomes, NUM_EVOLUTIONS)   # Run for set number of generations
    best_demo(winner, config, render_plot=True)   # Demonstrate the best performing genome
    print('\n\n\nBest genome:\n\n{!s}'.format(winner))  # show final stats

    if view_stats:
        # Create visualization.
        node_names = {-1: 'Cart_Pos', 
                      -2: 'Cart_Vel', 
                      -3:'Pole_Ang', 
                      -4:'Pole_Ang_vel',
                      0:'Action'}
        visualize.draw_net(config, winner, view=True, node_names=node_names, show_disabled=True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    return run_stats(stats)



def get_dict_average(data, render_plot=False):
    ''' 
        Return average of each element in dictionary
        and probability of each node being used
    '''
    data_dict = {i:[] for i in data[0].keys()}
    data_dict['weights'] = dict((k, []) for k in range(-NUM_INPUTS, 0))
    node_use_prob = dict((k, 0) for k in range(-NUM_INPUTS, 0))

    # aggregate all data to one dictionary 
    for d in data:
        for k, v in d.items():
            if k in data_dict and k in d:
                    if k != 'weights':
                        data_dict[k].append(v)
                    else:
                        for num, val in v.items():
                            for w in val:                            
                                data_dict['weights'][num].append(w)
                                node_use_prob[num] = node_use_prob[num] + 1

    # Compute average for items in dictionary
    ave_dict = copy.deepcopy(data_dict)

    for k, v in ave_dict.items():
        if k != 'weights':
            ave_dict[k] = sum(v)/len(v)
        else:
            for num, val in v.items():
                if len(val) != 0:
                    ave_dict['weights'][num] = sum(val)/len(val)

    # Get the probability of input node used for the network
    for k, v in node_use_prob.items():
        node_use_prob[k] = v / NUM_EXPERIMENTS

    if render_plot:
        fig, ax = plt.subplots()
        plt.title('Generations')
        ax.hist(data_dict['generations'], linewidth=0.1)
        plt.show()

    return ave_dict, node_use_prob



def expriment_csv(data):
    '''
        Export given dictionary data as CSV file 
    '''
    with open('./experiment_stats/Cart-pole_{}_experiments_{}.csv'.format(NUM_EXPERIMENTS, datetime.datetime.now()), 
              'w', encoding='UTF8', newline='') as f:
        fieldnames=list(data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in data:
            writer.writerow(d)   



if __name__ == '__main__':
    # Load neat-python configuration file 
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, 'neat_configs/cart-pole-config.txt')
    count = 0
    experiments_res = []

    for _ in range(NUM_EXPERIMENTS):
        data = run(config_path, view_stats=True)
        experiments_res.append(data)
        print('====== Experiments:', count, ' =============================================',)
        count += 1

    expriment_csv(experiments_res)
    env.close()