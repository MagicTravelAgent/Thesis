from scipy.stats import entropy
import numpy as np


def quality(pred_state, pred_outcome_given_state, expected_outcome):
    # Change lists to arrays
    pred_state = np.array(pred_state)
    pred_outcome_given_state = np.array(pred_outcome_given_state)
    expected_outcome = np.array(expected_outcome)

    # Determine predicted outcome
    pred_outcome = np.dot(pred_state, pred_outcome_given_state)

    # Determine the extrinsic and epistemic value
    extrinsic = np.sum(pred_outcome * np.log(expected_outcome))
    epistemic = epistemic_value(pred_state, pred_outcome_given_state, pred_outcome)

    return extrinsic + epistemic


def epistemic_value(pred_state, likelihoods, pred_outcome):
    # Calculate the posterior for each possible observation
    posterior = np.multiply(pred_state, likelihoods.T)
    post_sum = np.sum(posterior, axis=1)
    posterior = posterior / post_sum[:, None]

    # Calculate the expected entropy
    pred = pred_state * np.ones(posterior.shape)
    exp_ent = np.sum(pred_outcome * entropy(qk=pred, pk=posterior, axis=1))
    return exp_ent

#figure 6
def quality_alt(pred_state, pred_outcome_given_state, expected_outcome):
    # Change lists to arrays
    pred_state = np.array(pred_state)
    print("\npredicted state:", pred_state)
    pred_outcome_given_state = np.array(pred_outcome_given_state)
    print("\npredicted outcome given state:\n", pred_outcome_given_state)
    expected_outcome = np.array(expected_outcome)
    print("\nexpected outcome:", expected_outcome)

    # Determine predicted outcome
    pred_outcome = np.dot(pred_state, pred_outcome_given_state)
    print("\npredicted outcome:", pred_outcome)

    # Calculate predicted uncertainty as the expectation
    # of the entropy of the outcome, weighted by the
    # probability of that outcome
    pred_ent = np.sum(pred_state * entropy(pred_outcome_given_state, axis=1))
    print("\npredicted entropy:", pred_ent)

    # Calculate predicted divergence as the Kullback-Leibler
    # divergence between the predicted outcome and the expected outcome
    pred_div = entropy(pk=pred_outcome, qk=expected_outcome)
    print("\npredicted divergence:", pred_div)

    # Return the sum of the negatives of these two
    return -pred_ent-pred_div
    



# Define the distribution over future states P(s_t | pi)
pred_state = [0.5, 0.1, 0.4]

# Define the distribution over observation given a state P(o_t | s_t) 
pred_outcome_given_state = [[0.55, 0.15, 0.3],
                            [0.5, 0.49, 0.01],
                            [0.3, 0.6, 0.1]]

# Define the exepected out P(o | m)
expected_outcome = [0.49, 0.01, 0.5]


# There are two ways to determine the quality of a policy
q1 = quality(pred_state, pred_outcome_given_state, expected_outcome)
q2 = quality_alt(pred_state, pred_outcome_given_state, expected_outcome)

# Show that these are the same
print(q1, q2, q1==q2)