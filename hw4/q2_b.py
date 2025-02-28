import numpy as np

def monte_carlo_system_reliability(n_simulations=1000, failure_prob=0.35):
    """
    Monte Carlo simulation to estimate system reliability.
    
    Parameters:
    - n_simulations (int): Number of Monte Carlo iterations
    - failure_prob (float): Probability of failure for each component
    
    Returns:
    - failure_probability (float): Estimated probability of system failure
    - standard_deviation (float): Standard deviation of the estimate
    - coefficient_of_variation (float): Coefficient of variation of the estimate
    """
    np.random.seed(42)  # For reproducibility
    num_components = 6  # Total components in the system
    system_failures = np.zeros(n_simulations)  # Store failure outcomes

    for i in range(n_simulations):
        # Randomly assign failure states (1: working, 0: failed)
        states = np.random.rand(num_components) > failure_prob
        
        # Define minimum link-sets: {C1}, {C2}, {C3}, {C4, C5, C6}
        link_set_1 = states[0]  # C1
        link_set_2 = states[1]  # C2
        link_set_3 = states[2]  # C3
        link_set_4 = states[3] * states[4] * states[5]  # C4 AND C5 AND C6

        # The system works if at least one link-set is functional
        system_failures[i] = not (link_set_1 or link_set_2 or link_set_3 or link_set_4)

    # Compute failure probability and statistical measures
    failure_probability = np.mean(system_failures)
    standard_deviation = np.std(system_failures, ddof=1)
    coefficient_of_variation = standard_deviation / failure_probability if failure_probability > 0 else 0

    return failure_probability, standard_deviation, coefficient_of_variation

# Run the Monte Carlo simulation
failure_prob, std_dev, coef_var = monte_carlo_system_reliability()
print(f"Estimated Failure Probability: {failure_prob:.5f}")
print(f"Standard Deviation: {std_dev:.5f}")
print(f"Coefficient of Variation: {coef_var:.5f}")
