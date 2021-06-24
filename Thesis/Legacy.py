import numpy as np
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt
import pgmpy
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
  
def new():
    world = BayesianModel([('light', 'food'), ('prev', 'light'), ('action', 'light')])
    cpd_a = TabularCPD('action', 2, [[0.5], [0.5]])
    cpd_p = TabularCPD('prev', 2, [[0.5], [0.5]])
    cpd_l = TabularCPD('light', 2, [[0.99, 0.01, 0.01, 0.99], 
                                    [0.01, 0.99, 0.99, 0.01]],
                       ['prev', 'action'], [2, 2])
    cpd_f = TabularCPD('food', 3, [[0.6, 0.05], 
                                    [0.4, 0.475], 
                                    [0, 0.475]],
                   ['light'], [2])
    world.add_cpds(cpd_l, cpd_p, cpd_f, cpd_a)
    '''
    print(world.get_cpds('light'))
    infer = VariableElimination(world)
    print(infer.query(['food'], evidence={'prev': 0}))
    '''
    inference = BayesianModelSampling(world)
    evidence = [State(var='prev', state=0), State(var='action', state=0)]
    
    #----------------make some samples of an agent that does not move
    samples_1 = []
    inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
    inf = inf[0]
    samples_1.append([inf[-2],inf[-1]])
    
    for i in range(1, 100):    
        evidence = [State(var='prev', state=samples_1[i-1][0]), State(var='action', state=0)]
        inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
        inf = inf[0]
        samples_1.append([inf[-2],inf[-1]])
        
    samples_1 = np.array(samples_1)
    
    #----------------make some samples of an agent that moves randomly
    samples_2 = []
    evidence = [State(var='prev', state=0), State(var='action', state=0)]
    inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
    inf = inf[0]
    samples_2.append([inf[-2],inf[-1]])
    
    for i in range(1, 100):    
        evidence = [State(var='prev', state=samples_2[i-1][0]), State(var='action', state=np.random.randint(2))]
        inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
        inf = inf[0]
        samples_2.append([inf[-2],inf[-1]])
        
    samples_2 = np.array(samples_2)
    
    #----------------make some samples of an agent that has some intelligence or something
    samples_3 = []
    evidence = [State(var='prev', state=0), State(var='action', state=0)]
    
    #generate 3 samples at the start
    for i in range(3):
        inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
        inf = inf[0]
        samples_3.append([inf[-2],inf[-1]])
    
    for i in range(3, 100):
        act = action(np.array(samples_3)[:,1])
        evidence = [State(var='prev', state=samples_3[i-1][0]), State(var='action', state=act)]
        inf = inference.rejection_sample(evidence=evidence, size=1, return_type='list')
        inf = inf[0]
        samples_3.append([inf[-2],inf[-1]])
        
    samples_3 = np.array(samples_3)
    
    
    #----------------plot the agents
    sns.scatterplot(range(len(samples_1)),samples_1[:,1])
    for i in range(1,(len(samples_1))):
        if samples_1[i,0] != samples_1[i-1,0]:
            plt.axvline(i, color = 'r')
    plt.title("scatter graph showing samples over time for an agent that does not move")
    plt.show()
    print(np.mean(samples_1[:,1]))
    
    sns.scatterplot(range(len(samples_2)),samples_2[:,1])
    for i in range(1,(len(samples_2))):
        if samples_2[i,0] != samples_2[i-1,0]:
            plt.axvline(i, color = 'r')
    plt.title("scatter graph showing samples over time for an agent that moves randomly")
    plt.show()
    print(np.mean(samples_2[:,1]))
    
    sns.scatterplot(range(len(samples_3)),samples_3[:,1])
    for i in range(1,(len(samples_3))):
        if samples_3[i,0] != samples_3[i-1,0]:
            plt.axvline(i, color = 'r')
    plt.title("scatter graph showing samples for an agent that moves based on the last 3 samples")
    plt.show()
    print(np.mean(samples_3[:,1]))
    
def action(samples):
    last = samples[-3:]
    if (np.mean(last) < 0.34):
        return 1
    else:
        return 0

#the old agent that updated using entropy???
class EntropyAgent:
    def __init__(self):
        #init priors
        self.mod_1 = []
        self.mod_2 = []
        
        [self.mod_1.append(1) for i in range(40)]
        [self.mod_1.append(0) for i in range(60)]
        self.mod_1.append(2)
        
        [self.mod_2.append(1) for i in range(40)]
        [self.mod_2.append(2) for i in range(60)]
        self.mod_2.append(0)
        
        self.mods = [self.mod_1, self.mod_2]
        
        
        self.world = BayesianModel([('prev', 'pred state given action'), ('action', 'pred state given action'), ('pred state given action', 'exp outcome given state')])
        self.cpd_a = TabularCPD('action', 2, [[0.5], [0.5]])
        
        self.cpd_p = TabularCPD('prev', 2, [[0.5], [0.5]])
        
        self.cpd_psa = TabularCPD('pred state given action', 2, 
                                  [[0.99, 0.01, 0.01, 0.99], 
                                  [0.01, 0.99, 0.99, 0.01]],
                                  ['prev', 'action'], [2, 2])
        
        self.cpd_os = TabularCPD('exp outcome given state', 3,  
                                             [[0.6, 0], 
                                            [0.4, 0.4], 
                                            [0, 0.6]],
                                            ['pred state given action'], [2])

        self.cpd_e = TabularCPD('expected ', 3, [[0.001], [0.499], [0.5]])
        # probably the way that i can update the expected is by making some huge dirichlet for 
        # expected so that it is very certain and by the time the model knows what it wants it will not effect
        # the expected, but in the complete dark room it will be fucked because it will always go down
        
        self.world.add_cpds(self.cpd_p, self.cpd_a, self.cpd_psa, self.cpd_os)
        self.inferer = VariableElimination(self.world)

    def infer(self, prev, action):
        query = (self.inferer.query(['exp outcome given state'], evidence={'prev': prev, 'action': action}))
        return query
    
    def quality_alt(self, prev_state, action, prints = False):
        pred_state = np.array(self.inferer.query(['pred state given action'], evidence={'prev': prev_state, 'action': action}).values)
        
        pred_outcome_given_state = np.array(self.cpd_os.values)
        
        expected_outcome = np.array(self.cpd_e.values)
        
        # Determine predicted outcome
        pred_outcome = np.dot(pred_state, pred_outcome_given_state.T)
        
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
    
    
    #this gets the alpha for the dirichlet dist (which is just the sum of each item)
    def get_alpha(self, lst_in):
        unique, counts = np.unique(lst_in, return_counts=True)
        samp_dic = dict(zip(unique, counts))
        alpha = np.array([x for x in samp_dic.values()])
        return alpha
    
    #sees which dirichlet the new observation belongs to best and adds it there
    def best_model(self, samp):
        #needing to make copies of the model because otherwise it gets fucky
        in_1 = self.mods[0].copy()
        in_1.append(samp)
        out_1 = dirichlet.entropy(self.get_alpha(in_1))
        
        in_2 = self.mods[1].copy()
        in_2.append(samp)
        out_2 = dirichlet.entropy(self.get_alpha(in_2))
        
        #returning the index of the model that is better
        out = [out_1, out_2]
        best = out.index(min(out))
        self.mods[best].append(samp)
        self.update_os()
        return best
    
    #should run this code after an observation as it updates the predicted outcome of each state
    def update_os(self):
        print(self.get_alpha(self.mods[0]))
        s_1 = dirichlet.mean(self.get_alpha(self.mods[0]))
        s_2 = dirichlet.mean(self.get_alpha(self.mods[1]))
        new_s = list(np.array([s_1, s_2]).T)
        
        self.world.remove_cpds(self.cpd_os)

        self.cpd_os = TabularCPD('exp outcome given state', 3,  
                                     new_s,
                                    ['pred state given action'], [2])
        self.world.add_cpds(self.cpd_os)

new()