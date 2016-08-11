import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator as op

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        # Parameters in the Q - Learning algorithm
        self.alpha = 0.5  # learning rate (alpha)
        self.gamma = 0.8  # discount factor(gamma)
        self.eps = 0.5  # exploration rate(epsilon)
        self.action_0 = None  # Action in t-1
        self.reward_0 = None  # Reward in t-1
        self.state_0 = None  # State in t-1
        self.Q = {}  # Empty Q table

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.action_0 = None
        self.reward_0 = None
        self.state_0 = None
        # self.Q = {} # Learn traffic rules again after reset or not?

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state (current state)
        self.state = (tuple(inputs.values()), self.next_waypoint)
        print 'Current state:', self.state

        # Create zero-initialized state entry if current state doesn't exist in Q table
        if not self.Q.has_key(self.state):
            self.Q[self.state] = dict([(action, 0.0) for action in self.actions])
        print 'Q table entry for current state:', self.Q[self.state]
        #print 'Q table:', self.Q

        # TODO: Select action according to your policy
        # Find actions with max q-values for current state and randomly pick one of it if more than one.
        best_actions = [act for act, q_value in self.Q[self.state].iteritems() if q_value == max(self.Q[self.state].values())]
        action = best_actions[random.randint(0, len(best_actions)-1)]
        print 'Action to perform:', action, 'from best candidates:', best_actions

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Perform the calculation delayed by 1 step. Skip for first iteration (t=0).
        if not (None in [self.state_0, self.action_0, self.reward_0]):
            self.Q[self.state_0][self.action_0] = (1 - self.alpha) * self.Q[self.state_0][self.action_0] + self.alpha *\
            (self.reward_0 + self.gamma * max(self.Q[self.state].iteritems(), key=op.itemgetter(1))[1])

        # Save current state, action and reward for next step
        self.state_0 = self.state
        self.action_0 = action
        self.reward_0 = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, next_waypoint = {} reward = {}".format(deadline, inputs, action, self.next_waypoint, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
