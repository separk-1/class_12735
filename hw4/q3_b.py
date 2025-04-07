import numpy as np

def lognormal_params_from_mean_cv(mean, cv):
    """
    Compute the internal normal distribution parameters (mu, sigma) 
    for a log-normal distribution with given mean and coefficient of variation (COV).
    
    Given X ~ Lognormal(mu, sigma^2):
      mean(X) = exp(mu + sigma^2/2) = mean
      var(X)  = [exp(sigma^2) - 1] * exp(2mu + sigma^2)
      => COV(X) = sqrt(var(X)) / mean(X) = cv
    """
    sigma_sq = np.log(1.0 + cv**2)     # sigma^2
    sigma = np.sqrt(sigma_sq)
    # mean(X) = exp(mu + sigma^2/2) = mean => mu + sigma^2/2 = ln(mean)
    # => mu = ln(mean) - sigma^2/2
    mu = np.log(mean) - 0.5*sigma_sq
    return mu, sigma

def build_corr_matrix(n, rho):
    """
    Generate an n x n correlation matrix:
      Diagonal = 1, Off-diagonal = rho.
    """
    M = np.full((n, n), rho)
    np.fill_diagonal(M, 1.0)
    return M

def multivariate_lognormal(n_sims, n_dim, mu_ln, sigma_ln, rho, rng):
    """
    Generate n_sims samples of an n_dim-dimensional log-normal distribution
    with correlation rho.
    
    - Internal normal distribution Z ~ N(mu_vector, Cov)
    - mu_vector = [mu_ln, mu_ln, ..., mu_ln] (length n_dim)
    - Cov = sigma_ln^2 * corr_matrix(rho)
    - Returns an array of shape (n_sims, n_dim)
    """
    mean_vec = np.full(n_dim, mu_ln)
    
    # Covariance matrix = sigma_ln^2 * correlation matrix
    corr_mat = build_corr_matrix(n_dim, rho)
    cov_mat = sigma_ln**2 * corr_mat
    
    # Generate multivariate normal samples (shape=(n_sims, n_dim))
    Z = rng.multivariate_normal(mean_vec, cov_mat, size=n_sims)
    return np.exp(Z)  # Log-normal => exp(Z)

def simulate_bridge_system(n_simulations=10_000, max_bridges=10, budget=20.0):
    """
    Monte Carlo simulation:
    - Vary the number of bridges (1 to max_bridges) while ensuring that
      the mean capacity (mu_c) stays within the budget.
    - Demand: mean=5, CV=0.6, correlation coefficient=0.4
    - Capacity: mean=mu_c (determined in the code), CV=0.5, correlation coefficient=0.3
    - Cost function: C = C0 + kc * (mu_c / mu_0)^2
    - Estimate system failure probability (all bridges fail: demand > capacity)
    """
    rng = np.random.default_rng(42)  # For reproducibility

    mu_0 = 5.0   # Reference acceleration in m/s^2
    C0 = 2.0     # Fixed construction cost per bridge (M$)
    kc = 1.0     # Cost parameter
    # Demand parameters (mean=5, CV=0.6) => Log-normal internal normal mu, sigma
    demand_mean = 5.0
    demand_cv   = 0.6
    rho_d       = 0

    mu_ln_d, sigma_ln_d = lognormal_params_from_mean_cv(demand_mean, demand_cv)

    # Capacity correlation coefficient
    rho_c = 0.3
    capacity_cv = 0.5  # Given in the problem statement

    results = []

    def cost_function(mu_c):
        # Cost function from the problem: C = C0 + kc*(mu_c / mu_0)^2
        return C0 + kc*((mu_c / mu_0)**2)

    for n_bridges in range(1, max_bridges+1):
        # Increment mu_c by 0.1 until the total cost exceeds the budget
        step = 0.1
        test_mu_c = 0.0
        while True:
            if n_bridges * cost_function(test_mu_c) > budget:
                # Exceeds budget
                test_mu_c -= step
                break
            test_mu_c += step

        if test_mu_c < 0:
            # If even one bridge exceeds the budget, stop
            continue

        # Final determined mu_c
        mu_c = test_mu_c

        # Total cost
        total_cost = n_bridges * cost_function(mu_c)
        # Capacity distribution parameters (mean=mu_c, CV=0.5)
        mu_ln_c, sigma_ln_c = lognormal_params_from_mean_cv(mu_c, capacity_cv)

        # Monte Carlo simulation
        # demand shape=(n_simulations, n_bridges)
        demand_samples = multivariate_lognormal(n_simulations, n_bridges,
                                                mu_ln_d, sigma_ln_d,
                                                rho_d, rng)
        # capacity shape=(n_simulations, n_bridges)
        capacity_samples = multivariate_lognormal(n_simulations, n_bridges,
                                                  mu_ln_c, sigma_ln_c,
                                                  rho_c, rng)

        # System failure => All bridges experience demand_i > capacity_i
        fail_mask = np.all(demand_samples > capacity_samples, axis=1)
        failure_probability = np.mean(fail_mask)

        results.append((n_bridges, mu_c, total_cost, failure_probability))

    return results

# ===== Execution =====
if __name__ == "__main__":
    results = simulate_bridge_system()
    # Find configuration with lowest failure probability
    optimal_config = min(results, key=lambda x: x[3])

    print(f"Optimal Number of Bridges: {optimal_config[0]}")
    print(f"Mean Capacity: {optimal_config[1]:.2f} (m/s^2)")
    print(f"Total Cost: {optimal_config[2]:.2f} M$")
    print(f"System Failure Probability: {optimal_config[3]:.6f}")

    print("\nAll Configurations:")
    for n_bridges, mu_c, cost, p_fail in results:
        print(f"Bridges={n_bridges}, mu_c={mu_c:.2f}, "
              f"Cost={cost:.2f}M, P_fail={p_fail:.6f}")
