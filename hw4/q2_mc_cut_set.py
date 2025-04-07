import numpy as np

def monte_carlo_system_failure_cutsets(n_simulations=1000, failure_prob=0.35):
    """
    Monte Carlo simulation to estimate system failure
    using the given MINIMUM CUT-SETS:
      - {C1, C2, C3, C4}
      - {C1, C2, C3, C5}
      - {C1, C2, C3, C6}
    
    Parameters:
    -----------
    n_simulations : int
        Number of Monte Carlo iterations.
    failure_prob : float
        Probability of failure for each component.

    Returns:
    --------
    system_fail_prob : float
        Estimated probability of system failure.
    std_dev : float
        Standard deviation of the above estimator.
    coef_var : float
        Coefficient of variation (std / mean) of the system failure estimator.
    p_cut1 : float
        Probability that cut-set #1 (C1,C2,C3,C4) all fail in a simulation.
    p_cut2 : float
        Probability that cut-set #2 (C1,C2,C3,C5) all fail.
    p_cut3 : float
        Probability that cut-set #3 (C1,C2,C3,C6) all fail.
    p_cut1_and_2 : float
        Probability that BOTH cut-set #1 and #2 fail simultaneously.
    p_cut1_or_2 : float
        Probability of the union (cut-set #1 OR #2). 
        Should match p_cut1 + p_cut2 - p_cut1_and_2.
    """
    np.random.seed(42)  # for reproducibility
    
    # Arrays to store (0 or 1) indicators
    cut_1_fail = np.zeros(n_simulations)
    cut_2_fail = np.zeros(n_simulations)
    cut_3_fail = np.zeros(n_simulations)
    system_fail = np.zeros(n_simulations)  # system failure indicator

    for i in range(n_simulations):
        # Draw random states: True=component works, False=component fails
        # => "component fails" occurs with probability = failure_prob
        states = np.random.rand(6) > failure_prob
        
        # Define each cut-set event = "All these components fail" → 1, otherwise 0
        # states[j] = True (working), False (failed), so
        # (1 - states[j]) = 1 means failure, 0 means working.
        
        # Cut-set #1: C1, C2, C3, C4 all fail
        cut_1_fail[i] = (1 - states[0]) * (1 - states[1]) \
                        * (1 - states[2]) * (1 - states[3])
        
        # Cut-set #2: C1, C2, C3, C5 all fail
        cut_2_fail[i] = (1 - states[0]) * (1 - states[1]) \
                        * (1 - states[2]) * (1 - states[4])
        
        # Cut-set #3: C1, C2, C3, C6 all fail
        cut_3_fail[i] = (1 - states[0]) * (1 - states[1]) \
                        * (1 - states[2]) * (1 - states[5])
        
        # System failure (if at least one cut-set fails) is computed using OR
        # => OR(A,B,C) = 1 - (1-A)*(1-B)*(1-C)
        system_fail[i] = 1 - (1 - cut_1_fail[i]) \
                           * (1 - cut_2_fail[i]) \
                           * (1 - cut_3_fail[i])

    # ===== Compute statistics =====
    # 1) System failure probability
    system_fail_prob = np.mean(system_fail)
    std_dev = np.std(system_fail, ddof=1)
    coef_var = std_dev / system_fail_prob if system_fail_prob > 0 else 0

    # 2) Failure probability of each individual cut-set
    p_cut1 = np.mean(cut_1_fail)
    p_cut2 = np.mean(cut_2_fail)
    p_cut3 = np.mean(cut_3_fail)

    # 3) Intersection probability: both cut-set #1 and #2 fail simultaneously
    p_cut1_and_2 = np.mean(cut_1_fail * cut_2_fail)  # Proportion where both are 1
    
    # 4) Union probability of two cut-sets (using direct formula)
    #    P(A ∪ B) = P(A) + P(B) - P(A∩B)
    p_cut1_or_2 = p_cut1 + p_cut2 - p_cut1_and_2

    return (system_fail_prob, std_dev, coef_var,
            p_cut1, p_cut2, p_cut3,
            p_cut1_and_2, p_cut1_or_2)


# ===== Example execution =====
if __name__ == "__main__":
    (fail_prob, std_dev, cov,
     p_cut1, p_cut2, p_cut3,
     p_cut1_and_2, p_cut1_or_2) = monte_carlo_system_failure_cutsets(
                                         n_simulations=10_000,
                                         failure_prob=0.35
                                     )
    print(f"[System Failure via Cut-sets]")
    print(f"Estimated Failure Probability: {fail_prob:.6f}")
    print(f"Standard Deviation:           {std_dev:.6f}")
    print(f"Coefficient of Variation:     {cov:.6f}\n")

    print(f"P(Cut#1 fails) = {p_cut1:.6f}")
    print(f"P(Cut#2 fails) = {p_cut2:.6f}")
    print(f"P(Cut#3 fails) = {p_cut3:.6f}\n")

    print(f"P(Cut#1 and Cut#2 both fail) = {p_cut1_and_2:.6f}")
    print(f"P(Cut#1 or  Cut#2)           = {p_cut1_or_2:.6f}")
    print("Check formula => p1 + p2 - p1∩p2 ≈ p1∨p2\n")
