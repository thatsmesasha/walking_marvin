import gym
import os, sys
import numpy as np
import random
import cPickle as pickle
import argparse

class Model:
    def __init__(self):
        self.weights = [np.zeros((120, 64)), np.zeros((64,64)), np.zeros((64,4))]

    def predict(self, history):
        out = np.expand_dims(history.flatten(), 0)
        out /= np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class Evolution:
    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.001):
        np.random.seed(0)
        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate


    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try


    def get_weights(self):
        return self.weights


    def run(self, iterations, print_step=10):
        for iteration in range(iterations):
            if iteration % print_step == 0:
                print('Training %d finished with average reward %f' % (iteration, self.get_reward(self.weights)))

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self._get_weights_try(self.weights, population[i])
                rewards[i] = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T

class Agent:
    def __init__(self):
        self.history_length     = 5
        self.population_size    = 20
        self.sigma              = 0.1
        self.learning_rate      = 0.01
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.0
        self.exploration_decay  = 0.000001
        self.episodes_train     = 1
        self.tmp                = 1

        self.env                = gym.make('Marvin-v0')
        self.model              = Model()
        self.evolution          = Evolution(self.model.get_weights(), self.get_reward, self.population_size, self.sigma, self.learning_rate)

    def predict(self, history):
        prediction = self.model.predict(np.array(history))
        return prediction

    def load(self, file):
        with open(file, 'rb') as weights:
            self.model.set_weights(pickle.load(weights))
        self.evolution.weights = self.model.get_weights()

    def save(self, file):
        with open(file, 'wb') as weights:
            pickle.dump(self.evolution.weights, weights)

    def run(self, episodes, render=True):
        self.model.set_weights(self.evolution.weights)
        for index_episode in range(episodes):
            total_reward = 0.
            state = self.env.reset()
            history = [state] * self.history_length
            done = False
            index_step = 0
            while not done and (index_step < 3000 or not render):
                if render:
                    self.env.render()
                action = self.predict(history)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                history = history[1:]
                history.append(state)
                index_step += 1
            print ("Episode %d finished after %d steps with total reward %.2f" % (index_episode, index_step, total_reward))

    def train(self, episodes):
        self.evolution.run(episodes, print_step=1)

    def get_reward(self, weights):
        total_reward = 0.
        self.model.set_weights(weights)
        for index_episode in range(self.episodes_train):
            state = self.env.reset()
            history = [state] * self.history_length
            done = False
            index_step = 0
            render = False
            if self.tmp == 10 * self.population_size:
                self.tmp = 1
                render = True
            else:
                self.tmp += 1
            while not done and index_step < 1600:
                if render: self.env.render()
                if self.exploration_rate > self.exploration_min:
                    self.exploration_rate -= self.exploration_decay
                if random.random() < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = self.predict(history)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                history = history[1:]
                history.append(state)
                index_step += 1
        return total_reward / self.episodes_train

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Teach Marvin how to walk.')
    parser.add_argument('-l', '--load', dest='load_weights',
                      help='Load weights for Marvin agent from FILE. Skip training process if this option is specified.',
                      metavar='FILE')
    parser.add_argument('-s', '--save', dest='save_weights',
                      help='Save weights to FILE after running the program.',
                      metavar='FILE')
    parser.add_argument('-w', '--walk', action='store_true', dest='only_walking',
                      help='Display only walking process.', default=False)
    args = parser.parse_args()
    agent = Agent()
    try:
        if args.load_weights:
            agent.load(args.load_weights)
            print "Successfully loaded weights."
        if not (args.only_walking or args.load_weights):
            print "Starting training process..."
            agent.train(250)
        print "Now Marvin can walk!"
        agent.run(100)
    except KeyboardInterrupt:
        print "\nExiting..."
    except Exception as e:
        print "An error occured: " + str(e)
    finally:
        if args.save_weights:
            agent.save(args.save_weights)
            print "Successfully saved weights."
