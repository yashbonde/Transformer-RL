'''
ppo.py

Has functions for making the learner network and return the required tensors

20.08.2019 - @yashbonde
'''

import tensorflow as tf
from model import policy_network
from utils import assign_a_to_b_ops

# dictionaries for optimizer to use
train_opt = {
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}

# ppo function
def ppo(config, inp_tensors):
    """
    Proximal policy optimization function which returns the required tensors
    (https://arxiv.org/abs/1707.06347) - John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
    Very clever algorithm, I upvote!

    :param config: config
    :param inp_tensors:
    :return:
    """
    lr = config.lr
    gamma = config.gamma
    eps_clip = config.eps_clip
    c1 = config.c1
    c2 = config.c2

    # get input tensors
    state, rewards, advantage, next_values = inp_tensors

    # make the two policies and get output tensors from it
    (curr_action_probs, curr_action), curr_values = policy_network(config, state, scope = 'current_policy')
    (old_action_probs, old_action), _ = policy_network(config, state, scope = 'old_policy')

    # find variables
    pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'current_policy')
    pi_old_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'old_policy')
    assign_ops = assign_a_to_b_ops(pi_vars, pi_old_vars)

    # probabilities of actions which agent took with policy
    act_probs = curr_action_probs * tf.one_hot(indices=curr_action, depth=curr_action_probs.shape[1])
    curr_action_probs = tf.reduce_sum(act_probs, axis=1)

    # probabilities of actions which agent took with policy
    act_probs = old_action_probs * tf.one_hot(indices=old_action, depth=old_action_probs.shape[1])
    old_act_probs = tf.reduce_sum(act_probs, axis=1)

    # loss
    with tf.variable_scope('loss/clip'):
        ratios = tf.exp(tf.log(curr_action_probs) - tf.log(old_act_probs)) # so fancy
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 1 - eps_clip, clip_value_max = 1 + eps_clip)
        loss_clip = tf.minimum(advantage * ratios, advantage * clipped_ratios)
        loss_clip = tf.reduce_mean(loss_clip)
        tf.summary.scalar('loss_clip', loss_clip)

    # computation graph for values
    with tf.variable_scope('loss/value_func'):
        loss_vf = rewards + gamma * next_values
        loss_vf = tf.squared_difference(loss_vf, curr_values)
        loss_vf = tf.reduce_mean(loss_vf)
        tf.summary.scalar('loss_vf', loss_vf)

    # entropy
    with tf.variable_scope('loss/entropy'):
        entropy = tf.log(tf.clip_by_value(curr_action_probs, 1e-10, 1.0)) * curr_action_probs
        entropy = -tf.reduce_sum(entropy, axis = 1)
        entropy = tf.reduce_mean(entropy)
        tf.summary.scalar('entropy', entropy)

    # final loss combining the three
    with tf.variable_scope('loss'):
        loss = loss_clip - c1 * loss_vf  + c2 * entropy
        loss = -loss
        tf.summary.scalar('loss', loss)

    if config.opt in train_opt:
        train_step = train_opt[config.opt](lr).minimize(loss)
    else:
        raise ValueError('Optimizer can only be `adam`, `rmsprop` or `sgd`')

    return curr_action, curr_values, train_step, assign_ops

def train(config, env):
    """
    function to make and train a network
    :param config:
    :param env:
    :return:
    """
    pass