import numpy as np
import matplotlib.pyplot as plt

# This script analyzes how the system failure probability P_s changes with the number of subsystems (m).

# Objectives:
# - Compute the rate of change of the system failure probability, ∂P_s/∂m.
# - Evaluate the rate for P = 10%, n = 5, and m ranging from 3 to 8.
# - Plot the results using a log-scale for better visualization.

# Key Concepts:
# - In a series system, increasing the number of subsystems (m) increases the overall failure probability.
# - A larger value of ∂P_s/∂m indicates a greater impact of adding an additional subsystem on system reliability.
# - Using a log-scale plot helps visualize small changes more effectively.

# Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Given parameters
P = 0.1  # Component failure probability (10%)
n = 5    # Number of parallel components in each subsystem
m_values = np.arange(3, 9)  # m ranges from 3 to 8

# Function to compute system failure probability
def failure_prob(P, n, m):
    """ Computes the system failure probability P_s. """
    return 1 - (1 - P**n)**m

# Function to compute the rate of change dP_s/dm analytically
def dP_dm(P, n, m):
    """ Computes the rate of change of system failure probability with respect to m. """
    return -np.log(1 - P**n) * (1 - P**n)**m

# Compute the rate of change for each m
rates = np.array([dP_dm(P, n, m) for m in m_values])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(m_values, np.abs(rates), 'o-', label=r"$\left|\frac{\partial P_s}{\partial m}\right|$", 
         markersize=8, linewidth=2, color="#C41230")  # Line color changed to #C41230 (dark red)

# Set y-axis to log scale
plt.yscale("log")

# Labels and title
plt.xlabel("Number of Subsystems (m)", fontsize=14)
plt.ylabel("Rate of Change (Log Scale)", fontsize=14)
plt.title("Rate of Change of System Failure Probability w.r.t. m", fontsize=16)

# Improve tick marks and grid
plt.xticks(m_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

# Add legend
plt.legend(fontsize=12, loc="upper right")

# Adjust layout for better visualization
plt.tight_layout()

# Save and display the plot
plt.savefig("./results/q1_a_1.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
