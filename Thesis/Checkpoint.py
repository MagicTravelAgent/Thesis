import numpy as np
from scipy.stats import entropy
from scipy.stats import dirichlet
import seaborn as sns
import matplotlib.pyplot as plt
import pgmpy
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
import pandas as pd
import matplotlib


    
class Environment:
    def __init__(self):
        
        
        self.world = BayesianModel([('light', 'food'), ('prev', 'light'), ('action', 'light')])
        self.cpd_a = TabularCPD('action', 2, [[0.5], [0.5]])
        self.cpd_p = TabularCPD('prev', 2, [[0.5], [0.5]])
        self.cpd_l = TabularCPD('light', 2,     [[0.99, 0.01, 0.01, 0.99], 
                                                 [0.01, 0.99, 0.99, 0.01]],
                                        ['prev', 'action'], [2, 2])
        self.cpd_f = TabularCPD('food', 3,  [[0.8, 0.8], 
                                            [0.2, 0.2], 
                                            [0, 0]],
                       ['light'], [2])
        self.world.add_cpds(self.cpd_l, self.cpd_p, self.cpd_f, self.cpd_a)
        self.samples = [[0,0]]                                                                      #samples contains [light level, sample for that light level]
        self.inference = BayesianModelSampling(self.world)
    
    def sample(self,act):
        evidence = [State(var='prev', state=self.samples[-1][0]), State(var='action', state=act)]
        inf = self.inference.rejection_sample(evidence=evidence, size=1, return_type='list')
        inf = inf[0]
        self.samples.append([inf[-2],inf[-1]])
        return inf[-1]
    
    def graph(self):
        graph_samp = np.array(self.samples)
        sns.scatterplot(range(len(graph_samp)),graph_samp[:,1])
        for i in range(1,(len(graph_samp))):
            if graph_samp[i,0] != graph_samp[i-1,0]:
                plt.axvline(i-0.5, color = 'r')
            if process[i-1] == 1:
                plt.axvline(i-0.75, color = 'g')
        plt.title("Scatter graph showing samples over time")
        plt.xlabel("Epoch")
        plt.ylabel("Food Observed")  
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12,6.75)
#        fig.savefig('question2.png', dpi=100)
        plt.show()
        print(np.mean(graph_samp[:,1]))

class FristonAgent: 
    def __init__(self):
        #init priors
        self.mod_1 = []
        self.mod_2 = []
        
        [self.mod_1.append(1) for i in range(2)]
        [self.mod_1.append(0) for i in range(8)]
        self.mod_1.append(2)
        
        [self.mod_2.append(1) for i in range(2)]
        [self.mod_2.append(2) for i in range(8)]
        self.mod_2.append(0)
        
        self.mods = [self.mod_1, self.mod_2]
        
        self.prev_s = 0
        self.prev_a = 0
        
        self.world = BayesianModel([('prev', 'pred state given action'), ('action', 'pred state given action'), ('pred state given action', 'exp outcome given state')])
        self.cpd_a = TabularCPD('action', 2, [[0.5], [0.5]])
        
        self.cpd_p = TabularCPD('prev', 2, [[0.5], [0.5]])
        
        self.cpd_psa = TabularCPD('pred state given action', 2, 
                                  [[0.99, 0.01, 0.01, 0.99], 
                                  [0.01, 0.99, 0.99, 0.01]],
                                  ['prev', 'action'], [2, 2])
        
        self.cpd_os = TabularCPD('exp outcome given state', 3,  
                                             [[0.8, 0], 
                                            [0.2, 0.2], 
                                            [0, 0.8]],
                                            ['pred state given action'], [2])

        self.cpd_e = TabularCPD('expected ', 3, [[0.001], [0.499], [0.5]])
        # probably the way that i can update the expected is by making some huge dirichlet for 
        # expected so that it is very certain and by the time the model knows what it wants it will not effect
        # the expected, but in the complete dark room it will be fucked because it will always go down
        
        self.world.add_cpds(self.cpd_p, self.cpd_a, self.cpd_psa, self.cpd_os)
        self.inferer = VariableElimination(self.world)

    def infer(self, prev, action):
        query = self.inferer.query(['exp outcome given state'], evidence={'prev': prev, 'action': action})
        return query
    
    def quality_alt(self, prev_state, action, prints = False):
        pred_state = np.array(self.inferer.query(['pred state given action'], evidence={'prev': prev_state, 'action': action}).values)
        
        pred_outcome_given_state = np.array(self.cpd_os.values)
        
        expected_outcome = np.array(self.cpd_e.values)
        
        # Determine predicted outcome
        pred_outcome = np.array(self.inferer.query(['exp outcome given state'], evidence={'prev': prev_state, 'action': action}).values)
        
        # Calculate predicted uncertainty as the expectation
        # of the entropy of the outcome, weighted by the
        # probability of that outcome
        pred_ent = np.sum(pred_state * entropy(pred_outcome_given_state.T, axis=1))
    
        # Calculate predicted divergence as the Kullback-Leibler
        # divergence between the predicted outcome and the expected outcome
        pred_div = entropy(pk=pred_outcome, qk=expected_outcome)
        
        #prints?
        if(prints):
            print("\npredicted state", pred_state)
            print("\npredicted outcome given state:\n", pred_outcome_given_state)
            print("\nexpected outcome:", expected_outcome)
            print("\npredicted outcome:", pred_outcome)
            print("\npredicted entropy:", pred_ent)
            print("\npredicted divergence:", pred_div)      
            print("------------------------------------------------------------------------")
        
        
        # Return the sum of the negatives of these two
        return -pred_ent-pred_div
    
    def choose_action(self, state):
        action_1 = self.quality_alt(state,0)
        action_2 = self.quality_alt(state,1)
        actions = [action_1, action_2]
        return actions.index(max(actions))
    
#        action_1 = self.quality_alt(state,0)
#        action_2 = self.quality_alt(state,1)
#                
#        action_sum = action_1 + action_2
#        actions = [action_2/action_sum, action_1/action_sum]
#        
#        action = np.random.choice(2, p = actions)
        
        return action
    
    #this gets the alpha for the dirichlet dist (which is just the sum of each item)
    def get_alpha(self, lst_in):
        unique, counts = np.unique(lst_in, return_counts=True)
        samp_dic = dict(zip(unique, counts))
        alpha = np.array([x for x in samp_dic.values()])
        return alpha
    
    def bayesian_inference(self, samp, state, prints = False):        
        #posterior = likelihood * prior / evidence
        
        #   prob of hyp given evidence = (prob of evidence given hyp * prob of hypothesis before evidence) / probabilty of evidence
        #   evidence = food number
        #   hypothesis = state number
        #   use the bayesian network of the model to estiamte these things
        
        prob_e_given_h = self.inferer.query(['exp outcome given state'], evidence={'pred state given action': state}).values[samp]
        prob_h = self.inferer.query(['pred state given action'], evidence={'prev': self.prev_s, 'action': self.prev_a}).values[state]        
        prob_e = self.inferer.query(['exp outcome given state'], evidence={'prev': self.prev_s, 'action': self.prev_a}).values[samp]
        prob_h_given_e = prob_e_given_h * prob_h / prob_e

        if prints:
            print("\n----------------------------------------- prob_e_given_h",prob_e_given_h)               
            print("----------------------------------------- prob_h",prob_h)
            print("----------------------------------------- prob_e",prob_e)
            print("----------------------------------------- prob_h_given_e",prob_h_given_e)
            print("")
        return prob_h_given_e
    

    def best_model(self, samp):
        #infer where I am based on the posterior
        possible = [self.bayesian_inference(samp, i) for i in range(2)]
        
        #setting the current predicted state for guessing the next action
        self.prev_s = possible.index(max(possible))
        
        #update the posterior with the observation
        self.mods[self.prev_s].append(samp)
        self.update_os()
        
        #return the state it thinks its in
        return self.prev_s
    
    #should run this code after an observation as it updates the predicted outcome of each state
    def update_os(self):
        s_1 = dirichlet.mean(self.get_alpha(self.mods[0]))
        s_2 = dirichlet.mean(self.get_alpha(self.mods[1]))
        new_s = list(np.array([s_1, s_2]).T)
        
        self.world.remove_cpds(self.cpd_os)

        self.cpd_os = TabularCPD('exp outcome given state', 3,  
                                     new_s,
                                    ['pred state given action'], [2])
        self.world.add_cpds(self.cpd_os)
        
    def think(self, observation):
        #choosing which state the agent thinks its in based on the sample
        state = self.best_model(observation)
       
        #choosing the action that is best for this state
        action = self.choose_action(state)
        
        return action
        
class ClarkAgent(FristonAgent):
    def __init__(self):
        super().__init__()
        self.expected = []
        [self.expected.append(1) for i in range(1000)]
        [self.expected.append(2) for i in range(4000)]
        self.expected.append(0)
        
        exp_in = dirichlet.mean(self.get_alpha(self.expected))
        self.cpd_e = TabularCPD('expected ', 3, [[exp_in[0]], [exp_in[1]], [exp_in[2]]])

    
    def best_model(self, samp):
        #infer where I am based on the posterior
        possible = [self.bayesian_inference(samp, i) for i in range(2)]
        
        #setting the current predicted state for guessing the next action
        self.prev_s = possible.index(max(possible))
        
        #update the posterior with the observation
        self.mods[self.prev_s].append(samp)
        self.update_os()
        self.update_e(samp)
        
        #return the state it thinks its in
        return self.prev_s 
    
    #updating the expected values using all the observations and creating a dirichlet of them
    def update_e(self, samp):
        self.expected.append(samp)
        exp_in = dirichlet.mean(self.get_alpha(self.expected))
        self.cpd_e = TabularCPD('expected ', 3, [[exp_in[0]], [exp_in[1]], [exp_in[2]]])

        
process = []
track_exp = []

env = Environment()
#clark = FristonAgent()
clark = ClarkAgent()

sample_taken = 0
for i in range(100):
    # getting an action for the sample taken
    action = clark.think(sample_taken)
    
    # getting a new sample based on the action
    sample_taken = env.sample(action)
    
    # keeping track of the actions so far, 1: swap, 0,:stay
    process.append(action)
    
    #keeping track of expected outcome from clark agent
    track_exp.append(clark.cpd_e.values)

env.graph()

print("Learned distributions for states")
print("State 1:", dirichlet.mean(clark.get_alpha(clark.mods[0])))
print("State 2:",dirichlet.mean(clark.get_alpha(clark.mods[1])))

track_exp = np.array(track_exp)
sns.lineplot(data = track_exp)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Line graph showing expected outcomes over time")
plt.xlabel("Iteration")
plt.ylabel("Expected Probability")
plt.show()

#print(agent.think(0))
