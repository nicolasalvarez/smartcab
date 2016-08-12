import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator as op
from analysis import Reporter
import math
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        # Parameters in the Q - Learning algorithm
        self.alpha = 0.5  # learning rate (alpha) 0 < alpha < 1
        self.gamma = 0.8  # discount factor(gamma) 0 < gamma < 1
        self.eps = 1  # exploration rate(epsilon) 0 < eps < 1
        self.action_0 = None  # Action in t-1
        self.reward_0 = None  # Reward in t-1
        self.state_0 = None  # State in t-1
        self.Q = {}  # Empty Q table
        self.experience = 1 # smartcab "experience", increases with time

        # Setup infraction metrics to report
        self.live_plot = False
        self.infractions_rep = Reporter(metrics=['invalid_moves'], live_plot=self.live_plot)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.action_0 = None
        self.reward_0 = None
        self.state_0 = None

    def update(self, t):

        # Alpha decay rate
        alpha_dr = self.experience
        # Eps decay rate
        eps_dr = math.log(self.experience + 2) # "+2" avoids div. by zero

        # Increase experience
        self.experience += 1

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state (current state)
        self.state = (tuple(inputs.values()), self.next_waypoint)
        #print 'Current state:', self.state

        # Create zero-initialized state entry if current state doesn't exist in Q table
        if not self.Q.has_key(self.state):
            self.Q[self.state] = dict([(action, 0.0) for action in self.actions])
        #print 'Q table entry for current state:', self.Q[self.state]
        # print 'Q table:', self.Q

        # TODO: Select action according to your policy
        # Find actions with max q-values for current state and randomly pick one of it if more than one.
        best_actions = [act for act, q_value in self.Q[self.state].iteritems() if
                        q_value == max(self.Q[self.state].values())]

        if self.eps / eps_dr < random.random():
            action = best_actions[random.randint(0, len(best_actions) - 1)]  # Select action using Q-table
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]  # Select action randomly
        #print 'Action to perform:', action, 'from best candidates:', best_actions

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.infractions_rep.collect('invalid_moves', t, reward)
        if self.live_plot:
            self.infractions_rep.refresh_plot()  # autoscales axes, draws stuff and flushes events

        # TODO: Learn policy based on state, action, reward
        # Perform the calculation delayed by 1 step. Skip for first iteration (t=0).
        if not (None in [self.state_0, self.action_0, self.reward_0]):
            self.Q[self.state_0][self.action_0] = (1 - self.alpha / alpha_dr) * \
                self.Q[self.state_0][self.action_0] + self.alpha / alpha_dr * \
                (self.reward_0 + self.gamma * max(self.Q[self.state].iteritems(), key=op.itemgetter(1))[1])

        # Save current state, action and reward for next step
        self.state_0 = self.state
        self.action_0 = action
        self.reward_0 = reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, next_waypoint = {} reward = {}".format(deadline, inputs, action, self.next_waypoint, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    n_trials = 100

    op_alpha = 0.0  # optimal learning rate
    op_gamma = 0.0  # optimal discount factor
    op_eps = 0.0  # optimal exploration rate

    # Setup metrics to report per simulation
    param_metrics = Reporter(metrics=['alpha', 'gamma', 'epsilon', 'trips_perf', 'infractions_perf'], live_plot=True)

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False,
                    live_plot=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    idx = 0
    eps_range = [0] #np.arange(0,0.2,0.05)
    gamma_range = [0.1] #np.arange(0,0.6,0.1)
    alpha_range = [0.4] #np.arange(0.2,0.6,0.05)
    for epsilon in eps_range:
        for gamma in gamma_range:
            for alpha in alpha_range:
                print "Running simulation for: ", epsilon, gamma, alpha
                # Run n_sims simulations for given parameters and store results in lists trips_perf and infractions_perf
                n_sims = 1
                trips_perf = []
                infractions_perf = []

                for i in range(n_sims):
                    # Set agent parameters and reset experience
                    a.alpha = alpha
                    a.gama = gamma
                    a.eps = epsilon
                    a.experience = 1
                    a.infractions_rep.reset()

                    sim.run(n_trials=n_trials)  # run for a specified number of trials
                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    # print sim.rep.summary()

                    for metric in sim.rep.summary():
                        if metric.name == 'success':
                            t_p = metric.sum() * 100 / n_trials
                            print "Name: {}, samples: {}, Performance: {}%".format(metric.name, len(metric), t_p)
                            trips_perf.append(t_p)

                    for metric in a.infractions_rep.summary():
                        if metric.name == 'invalid_moves':
                            i_p = abs(metric[metric == -1.0].sum()) * 100 / metric.size
                            print "Name: {}, samples: {}, Performance: {}%".format(metric.name, len(metric), i_p)
                            infractions_perf.append(i_p)

                # Collect metrics
                param_metrics.collect('alpha', idx, alpha)
                param_metrics.collect('gamma', idx, gamma)
                param_metrics.collect('epsilon', idx, epsilon)
                param_metrics.collect('trips_perf', idx, pd.Series(trips_perf).mean())
                param_metrics.collect('infractions_perf', idx, pd.Series(infractions_perf).mean())
                idx += 1

    # Show results
    results = pd.DataFrame(param_metrics.summary()).transpose()
    print results
    print 'Best configuration for trips performance:'
    print results.loc[results['trips_perf'].idxmax()]
    print 'Best configuration for traffic rules performance:'
    print results.loc[results['infractions_perf'].idxmin()]
    param_metrics.refresh_plot()
    param_metrics.show_plot()

if __name__ == '__main__':
    run()
