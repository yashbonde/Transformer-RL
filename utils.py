'''
file with util functions and classes

20.08.2019 - @yashbonde
'''

import copy
import tensorflow as tf


def save_config_to_json(config, save_path):
    """
    save the configuration to JSON and save at save path
    :param config:
    :param save_path:
    :return:
    """
    if not save_path[-5:] == '.json'
        save_path += '.json'
    pass


def calculate_advantage(rewards, values, values_next, gamma = 0.99):
    """
    calculate the GAE values from rewards, v and v'

    :param rewards:
    :param values:
    :param values_next:
    :param gamma:
    :return:
    """
    deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, values, values_next)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + gamma * gaes[t + 1]
    del deltas
    return gaes

def assign_a_to_b_ops(a, b):
    """
    Assign values from network A to network B
    :param a: tf.get_collection() for source
    :param b: tf.get_collection() for target
    :return: list of ops
    """
    with tf.variable_scope('assign_ops'):
        assign_ops = []
        for v, v_old in zip(a, b):
            assign_ops.append(tf.assign(v_old, v))
    return assign_ops

# class to handle memory --> reason I put this in class is that it removes
# the clusterfuck that scripts with many lists become, very difficult to read
class MemoryPPO:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.v_preds = []
        self.rewards = []

    def reset_memory(self):
        del self.observations[:]
        del self.actions[:]
        del self.v_preds[:]
        del self.rewards[:]

    def update(self, obs, act, v_pred, reward):
        self.observations.append(obs)
        self.actions.append(act)
        self.v_preds.append(v_pred)
        self.rewards.append(reward)