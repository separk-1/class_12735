import numpy as np

def simulate_bridge_system(n_simulations=10000, max_bridges=10, budget=20):
    """
    Monte Carlo simulation to find the optimal number of bridges
    that minimizes system failure probability within a given budget.
    
    Parameters:
    -----------
    n_simulations : int
        Number of Monte Carlo iterations.
    max_bridges : int
        Maximum number of bridges to consider.
    budget : float
        Total budget in million dollars.
    
    Returns:
    --------
    results : list
        A list of tuples containing (number of bridges, mean capacity, total cost, failure probability)
    """
    np.random.seed(42)  # For reproducibility
    
    mu_0 = 5  # Reference acceleration in ms^-2
    C0 = 2  # Fixed construction cost per bridge ($M)
    kc = 1  # Cost parameter ($M)
    
    demand_mean = 5  # Mean spectral acceleration demand in ms^-2
    demand_cv = 0.6  # Coefficient of variation of demand (60%)
    capacity_cv = 0.5  # Coefficient of variation of capacity (50%)
    rho_d = 0.4  # Correlation coefficient for demand
    rho_c = 0.3  # Correlation coefficient for capacity
    
    results = []
    
    for n_bridges in range(1, max_bridges + 1):
        # Solve for optimal mean capacity (mu_c) given cost constraint
        def cost_function(mu_c):
            return C0 + kc * (mu_c / mu_0) ** 2
        
        # Find the highest mu_c that keeps cost within budget
        mu_c = mu_0  # Start with reference capacity
        while n_bridges * cost_function(mu_c) <= budget:
            mu_c += 0.1  # Incrementally increase mean capacity
        mu_c -= 0.1  # Step back to last valid capacity
        
        # Compute total cost
        total_cost = n_bridges * cost_function(mu_c)
        
        # Generate Monte Carlo samples for demand and capacity
        demand_samples = np.exp(
            np.random.normal(np.log(demand_mean), np.sqrt(np.log(1 + demand_cv**2)), (n_simulations, n_bridges))
        )
        capacity_samples = np.exp(
            np.random.normal(np.log(mu_c), np.sqrt(np.log(1 + capacity_cv**2)), (n_simulations, n_bridges))
        )
        
        # Compute system failure (if all bridges fail simultaneously)
        system_failures = np.all(demand_samples > capacity_samples, axis=1)
        failure_probability = np.mean(system_failures)
        
        results.append((n_bridges, mu_c, total_cost, failure_probability))
    
    return results

# ===== Run Simulation and Find Optimal Configuration =====
if __name__ == "__main__":
    results = simulate_bridge_system()
    optimal_config = min(results, key=lambda x: x[3])  # Find config with lowest failure probability
    
    print(f"Optimal Number of Bridges: {optimal_config[0]}")
    print(f"Mean Capacity: {optimal_config[1]:.2f} ms^-2")
    print(f"Total Cost: {optimal_config[2]:.2f}M")
    print(f"System Failure Probability: {optimal_config[3]:.6f}")
    
    print("\nAll Investigated Configurations:")
    for n_bridges, mu_c, total_cost, fail_prob in results:
        print(f"Bridges: {n_bridges}, Mean Capacity: {mu_c:.2f}, Cost: {total_cost:.2f}M, Failure Probability: {fail_prob:.6f}")
