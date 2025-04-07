import numpy as np

# Define the transition matrix
P = np.array([
    [0.99, 0.01, 0.0],
    [0.0, 0.99, 0.01],
    [0.0, 0.0, 1.0]
])

# Initial state vector: starting from Intact
pi_0 = np.array([1.0, 0.0, 0.0])

# Function to compute state probabilities after t steps
def compute_state_prob(pi_0, P, t):
    P_t = np.linalg.matrix_power(P, t)
    return pi_0 @ P_t

# Compute state probabilities after 50 and 100 weeks
pi_50 = compute_state_prob(pi_0, P, 50)
pi_100 = compute_state_prob(pi_0, P, 100)

# Print the results
print(f"t = 50: Intact={pi_50[0]:.4f}, Damage={pi_50[1]:.4f}, Failure={pi_50[2]:.4f}")
print(f"t = 100: Intact={pi_100[0]:.4f}, Damage={pi_100[1]:.4f}, Failure={pi_100[2]:.4f}")