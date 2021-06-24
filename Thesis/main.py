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


    
class Environment:
    def __init__(self):
        
        
        self.world = BayesianModel([('light', 'food'), ('prev', 'light'), ('action', 'light')])
        self.cpd_a = TabularCPD('action', 2, [[0.5], [0.5]])
        self.cpd_p = TabularCPD('prev', 2, [[0.5], [0.5]])
        self.cpd_l = TabularCPD('light', 2,     [[0.99, 0.01, 0.01, 0.99], 
                                                 [0.01, 0.99, 0.99, 0.01]],
                                        ['prev', 'action'], [2, 2])
        self.cpd_f = TabularCPD('food', 3,  [[0.8, 0.0], 
                                            [0.2, 0.2], 
                                            [0, 0.8]],
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
        plt.title("Scatter graph showing samples over time \nRed: change of state \nGreen: agent chose to change")
        plt.show()
        print(np.mean(graph_samp[:,1]))
        
    def into_darkness(self):
        self.world.remove_cpds(self.cpd_f)
        new_cpd = np.array([[0.8, 0.2, 0],[0.8, 0.2, 0]]).T
        self.cpd_f = TabularCPD('food', 3, new_cpd, ['light'], [2])
        self.world.add_cpds(self.cpd_f)

class FristonAgent: 
    def __init__(self):
        #init priors
        self.mod_1 = [8,2,0.001]
        self.mod_2 = [0.001,2,8]
        
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

        self.cpd_e = TabularCPD('expected ', 3, [[0.001], [0.199], [0.8]])
        
        self.world.add_cpds(self.cpd_p, self.cpd_a, self.cpd_psa, self.cpd_os)
        
        self.update_os()
        
        self.inferer = VariableElimination(self.world)
        
        self.check_expectations = False

    def infer(self, prev, action):
        query = self.inferer.query(['exp outcome given state'], evidence={'prev': prev, 'action': action})
        return query
            
    def quality(self, prev_state, action, prints = False):
        # Change lists to arrays
        pred_state = np.array(self.inferer.query(['pred state given action'], evidence={'prev': prev_state, 'action': action}).values)
        
        pred_outcome_given_state = np.array(self.cpd_os.values).T
        
        expected_outcome = np.array(self.cpd_e.values)
    
        # Determine predicted outcome
        pred_outcome = np.dot(pred_state, pred_outcome_given_state)
    
        # Determine the extrinsic and epistemic value
        extrinsic = np.sum(pred_outcome * np.log(expected_outcome))
        epistemic = self.epistemic_value(pred_state, pred_outcome_given_state, pred_outcome)
        
        #prints?
        if(prints):
            print("\npredicted state", pred_state)
            print("\npredicted outcome given state:\n", pred_outcome_given_state)
            print("\nexpected outcome:", expected_outcome)
            print("\npredicted outcome:", pred_outcome)
            print("\extrinsic value:", extrinsic)
            print("\epistemic value:", epistemic)      
            print("------------------------------------------------------------------------")
    
        return extrinsic + epistemic
    
    def epistemic_value(self, pred_state, likelihoods, pred_outcome):
        # Calculate the posterior for each possible observation
        posterior = np.multiply(pred_state, likelihoods.T)
        post_sum = np.sum(posterior, axis=1)
        posterior = posterior / post_sum[:, None]
    
        # Calculate the expected entropy
        pred = pred_state * np.ones(posterior.shape)
        exp_ent = np.sum(pred_outcome * entropy(qk=pred, pk=posterior, axis=1))
        return exp_ent
    
    #calculate the the utility of each move and sample a move to take
    def choose_action(self, state):
        action_1 = self.quality(state,0)
        action_2 = self.quality(state,1)
                
        action_sum = action_1 + action_2
        actions = [action_2/action_sum, action_1/action_sum]
        
        action = np.random.choice(2, p = actions)
        return action

    
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
        self.mods[self.prev_s][samp] += 1
        self.update_os()
        
        #return the state it thinks its in
        return self.prev_s
    
    #should run this code after an observation as it updates the predicted outcome of each state
    def update_os(self):
        s_1 = dirichlet.mean(self.mods[0])
        s_2 = dirichlet.mean(self.mods[1])
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
        
        self.prev_a = action
        
        return action
        
class ClarkAgent(FristonAgent):
    def __init__(self, e1, e2, e3):
        super().__init__()
        self.expected = [e1, e2, e3]
        
        exp_in = dirichlet.mean(self.expected)
        self.cpd_e = TabularCPD('expected ', 3, [[exp_in[0]], [exp_in[1]], [exp_in[2]]])
        
        self.adapted = False

    
    def best_model(self, samp):
        #infer where I am based on the posterior
        possible = [self.bayesian_inference(samp, i) for i in range(2)]
        
        #setting the current predicted state for guessing the next action
        self.prev_s = possible.index(max(possible))
        
        #update the posterior with the observation
        self.mods[self.prev_s][samp] += 1
        self.update_os()
        self.update_e(samp)
        
        #return the state it thinks its in
        return self.prev_s
    
    #updating the expected values using all the observations and creating a dirichlet of them
    def update_e(self, samp):
        self.expected[samp] += 1
        exp_in = dirichlet.mean(self.expected)
        self.cpd_e = TabularCPD('expected ', 3, [[exp_in[0]], [exp_in[1]], [exp_in[2]]])
        
    #calculate the the utility of each move and sample a move to take
    def choose_action(self, state):
        action_1 = self.quality(state,0)
        action_2 = self.quality(state,1)
                
        action_sum = action_1 + action_2
        actions = [action_2/action_sum, action_1/action_sum]
        
        action = np.random.choice(2, p = actions)
        
        print(actions)
        if actions[0] > actions[1]:
            self.adapted = True
            print("adapted")
        else:
            self.adapted = False

        
        return action

def iterate(env, clark, process, track_exp, num_iter, sample_taken = 0):
    for i in range(num_iter):
        # getting an action for the sample taken
        action = clark.think(sample_taken)
        
        # getting a new sample based on the action
        sample_taken = env.sample(action)
        
        # keeping track of the actions so far, 1: swap, 0,:stay
        process.append(action)
        
        #keeping track of expected outcome from clark agent
        track_exp.append(clark.cpd_e.values)
    return sample_taken

def iterate_until_cracked(env, clark, process, track_exp, sample_taken = 0):
    i = 0
    count = 0
    while(count < 5):
        # getting an action for the sample taken
        action = clark.think(sample_taken)
        
        # checking to see if the agent gave up moving
        if clark.adapted:
            count = count + 1
        else:
            count = 0
        
        # getting a new sample based on the action
        sample_taken = env.sample(action)
        
        # keeping track of the actions so far, 1: swap, 0,:stay
        process.append(action)
        
        # keeping track of expected outcome from clark agent
        track_exp.append(clark.cpd_e.values)
        
        # counting the number of iterations until expectation changed
        i = i + 1
    return i

def graphing(env, clark, track_exp):
    env.graph()
    
    print("Learned distributions for states")
    print("State 1:", list(clark.cpd_os.values[:,0]))
    print("State 2:", list(clark.cpd_os.values[:,1]))
    
    track_exp = np.array(track_exp)
    sns.lineplot(data = track_exp)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Line graph showing expected outcomes over time")
    plt.xlabel("Iteration")
    plt.ylabel("Expected Probability")
    plt.show()
    
def experiment(priors, filename):
    x = [100]
    data_col = []
    
    for epoch in range(1):
        for i in x:
            # initialise the tracking lists as well as the environment
            process = []
            track_exp = []
            env = Environment()
            env.into_darkness()
            
            # selecting the agent you want to test
    #        clark = FristonAgent()
            clark = ClarkAgent(1,priors[0], priors[1])
            
            # run the experiment
            sample_taken = iterate(env, clark, process, track_exp, i)
#            sample_taken = iterate_until_cracked(env, clark, process, track_exp, )
            
    #        env.into_darkness()
    #        y = (iterate_until_cracked(sample_taken) - 5)
            data_col.append([epoch, sample_taken - 5, track_exp[-1]])
    
            graphing(env, clark, track_exp)
    
#    data_col = np.array(data_col)
#    data = {'epoch': data_col[:,0],
#            'time until adapted': data_col[:,1],
#            'expectations': data_col[:,2]
#            }
#    df = pd.DataFrame(data)
    
    #sns.lineplot(data = df, x = 'x', y = 'y')
    #plt.title("Line graph showing expected outcomes over time")
    #plt.xlabel("Prior Exploration")
    #plt.ylabel("Time Until Mentally Broken")
    #plt.show()
    
    df.to_csv(filename)

filenames = [
#"Clark_dark_data_10_40.csv",
#"Clark_dark_data_20_80.csv",
#"Clark_dark_data_30_120.csv",
#"Clark_dark_data_40_160.csv",
#"Clark_dark_data_50_200.csv",
#"Clark_dark_data_60_240.csv",
#"Clark_dark_data_70_280.csv",
#"Clark_dark_data_80_320.csv",
#"Clark_dark_data_90_360.csv",
#"Clark_dark_data_100_400.csv",
#"Clark_dark_data_110_440.csv",
#"Clark_dark_data_120_480.csv",
#"Clark_dark_data_200_800.csv",
#"Clark_dark_data_300_1200.csv",
#"Clark_dark_data_400_1600.csv",
"temp.csv"]

prior_list = [
[ 10 , 40 ],
[ 20 , 80 ],
[ 30 , 120 ],
[ 40 , 160 ],
[ 50 , 200 ],
[ 60 , 240 ],
[ 70 , 280 ],
[ 80 , 320 ],
[ 90 , 360 ],
[ 100 , 400 ],
[110, 440],
[120, 480],
[200, 800],
[300, 1200],
[400, 1600],
[500, 2000]]

[experiment(prior_list[i], filenames[i]) for i in range(len(filenames))]
# this function is used to see the graphs of an individual agent
#graphing(track_exp)

#env = Environment()
#print(env.cpd_l)































































