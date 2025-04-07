import numpy as np
import matplotlib.pyplot as plt

# This script analyzes how the system failure probability P_s changes with the number of parallel components (n).

# Objectives:
# - Compute the rate of change of the system failure probability, ∂P_s/∂n.
# - Evaluate the rate for P = 10%, m = 5, and n ranging from 3 to 8.
# - Plot the results using a log-scale for better visualization.

# Key Concepts:
# - In a parallel system, increasing the number of components (n) decreases the overall failure probability.
# - A larger value of ∂P_s/∂n indicates a greater impact of adding an additional redundant component.
# - Using a log-scale plot helps visualize small changes more effectively.

# Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Given parameters
P = 0.1  # Component failure probability (10%)
m = 5    # Number of subsystems
n_values = np.arange(3, 9)  # n ranges from 3 to 8

# Function to compute system failure probability
def failure_prob(P, n, m):
    """ Computes the system failure probability P_s. """
    return 1 - (1 - P**n)**m

# Function to compute the rate of change dP_s/dn analytically
def dP_dn(P, n, m):
    """ Computes the rate of change of system failure probability with respect to n. """
    return -m * (1 - P**n)**(m-1) * np.log(P) * P**n

# Compute the rate of change for each n
rates = np.array([dP_dn(P, n, m) for n in n_values])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, np.abs(rates), 'o-', label=r"$\left|\frac{\partial P_s}{\partial n}\right|$", 
         markersize=8, linewidth=2, color="#009647")  # Line color set to #C41230 (dark red)

# Set y-axis to log scale
plt.yscale("log")

# Labels and title
plt.xlabel("Number of Parallel Components (n)", fontsize=14)
plt.ylabel("Rate of Change (Log Scale)", fontsize=14)
plt.title("Rate of Change of System Failure Probability w.r.t. n", fontsize=16)

# Adjust grid for better readability
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)  # Major grid lines
plt.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)  # Minor grid lines (fainter)


# Improve tick marks and grid
plt.xticks(n_values, fontsize=12)
plt.yticks(fontsize=12)

# Add legend
plt.legend(fontsize=12, loc="upper right")

# Adjust layout for better visualization
plt.tight_layout()

# Save and display the plot
plt.savefig("./results/q1_a_2.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
