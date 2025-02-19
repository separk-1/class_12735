import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pandas as pd
from q1 import run_form_analysis, initialize_parameters

def limit_state_function(z):
    """ Computes the limit state function g(z) in the physical space. """
    return np.log(z[4]) - np.log(z[2]) - z[3] * np.log((z[0] + z[1]) / 900)

def monte_carlo_simulation(L, lambda_vec, n_samples):
    """
    Performs Monte Carlo (MC) simulation in standard normal space to estimate failure probability.

    Parameters:
    - L: Cholesky decomposition matrix
    - lambda_vec: Log-normal mean vector
    - n_samples: Number of Monte Carlo samples

    Returns:
    - Pf: Estimated probability of failure
    - std_dev: Standard deviation of the probability estimate
    - beta: Reliability index
    """
    # Generate standard normal samples
    u_samples = np.random.randn(5, n_samples)

    # Transform to x-space
    x_samples = L @ u_samples + lambda_vec

    # Transform to z-space
    z_samples = np.exp(x_samples)

    # Evaluate limit state function
    g_values = np.apply_along_axis(limit_state_function, 0, z_samples)

    # Compute failure probability (g < 0)
    failures = g_values < 0
    Pf = np.mean(failures)
    std_dev = np.sqrt((Pf * (1 - Pf)) / n_samples)  # Standard deviation of estimator

    # Compute reliability index β
    beta = -stats.norm.ppf(Pf) if Pf > 0 else np.inf  # Avoid log(0) issues

    return Pf, std_dev, beta

def importance_sampling(L, lambda_vec, u_star, n_samples, s = 1):
    """
    Performs Importance Sampling (IS) to estimate failure probability with varying s.

    Parameters:
    - L: Cholesky decomposition matrix
    - lambda_vec: Log-normal mean vector
    - u_star: Design point in standard normal space
    - n_samples: Number of Monte Carlo samples
    - s: Scaling factor for importance distribution

    Returns:
    - Pf: Estimated probability of failure
    - std_dev: Standard deviation of the probability estimate
    - beta: Reliability index
    """
    # Expand u_star to match sample size
    u_star_expanded = np.tile(u_star, (1, n_samples))

    # Generate samples from Importance Distribution N(u*, s^2 * I)
    u = s * np.random.randn(n_samples, 5) + u_star_expanded.T  

    # Transform to x-space
    lambda_vec_expanded = np.tile(lambda_vec, (1, n_samples))
    x_samples = np.dot(L, u.T) + lambda_vec_expanded

    # Transform to z-space
    z_samples = np.exp(x_samples)

    # Extract variables from z
    Fa, Fb, q, alpha, c = z_samples

    # Evaluate limit state function g(z)
    d = q * ((Fa + Fb) / 900) ** alpha
    failures = d > c  

    # Compute importance weights
    u_k = u.T  
    w_matrix = np.exp(
        -0.5 * (
            np.dot(u_k.T, u_k) - np.dot(
                np.dot((u_k - u_star_expanded).T, np.linalg.inv(np.diag([s] * 5))),
                (u_k - u_star_expanded)
            )
        )
    )
    w = np.diag(w_matrix)

    # Compute probability of failure using weighted sum
    Pf = np.sum(w * failures) / np.sum(w)

    # Compute standard deviation
    std_dev = np.sqrt(np.sum(w * ((failures - Pf) ** 2)) / np.sum(w)) / np.sqrt(n_samples)

    # Compute reliability index β
    beta = -stats.norm.ppf(Pf) if Pf > 0 else np.inf

    return Pf, std_dev, beta


if __name__ == "__main__":

    # Run FORM analysis to get u*
    results_df = run_form_analysis()
    u_star = np.array(results_df.iloc[0, -1]).reshape(-1,1)  # Extract last iteration result

    # Compute the design point in physical space
    lambda_vec, L = initialize_parameters()
    z_star = np.exp(L @ u_star + lambda_vec)

    print("Design Point z* in Physical Space:")
    print(np.round(z_star, 4))

    # Define different Monte Carlo sample sizes
    sample_sizes = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]

    # Store results in DataFrame
    mc_results = []

    for n in sample_sizes:
        Pf, std_dev, beta = monte_carlo_simulation(L, lambda_vec, n)
        mc_results.append([n, Pf, std_dev, beta])
        print(f"MC Simulations: {n} | Pf: {Pf:.5f} | Std Dev: {std_dev:.5f} | Beta: {beta:.4f}")

    # Convert results to DataFrame and save
    df_mc_results = pd.DataFrame(mc_results, columns=["Samples", "Pf", "Std Dev", "Beta"])
    df_mc_results.to_csv("./results/mc_simulation_results.csv", index=False)

    # Display the final table
    print(df_mc_results)

    # Store results in DataFrame for Importance Sampling
    is_results = []

    sample_sizes = [100, 300, 1000, 3000, 10000, 30000]
    for n in sample_sizes:
        Pf, std_dev, beta = importance_sampling(L, lambda_vec, u_star, n)
        is_results.append([n, Pf, std_dev, beta])
        print(f"Importance Sampling: {n} | Pf: {Pf:.5f} | Std Dev: {std_dev:.5f} | Beta: {beta:.4f}")

    # Convert results to DataFrame and save
    df_is_results = pd.DataFrame(is_results, columns=["Samples", "Pf", "Std Dev", "Beta"])
    df_is_results.to_csv("./results/is_simulation_results.csv", index=False)

    # Display the final table
    print(df_is_results)


    # Define different values of s and fixed n=1000
    s_values = [0.1, 0.3, 1, 3, 10]
    n_samples = 1000

    # Store results in DataFrame
    is_results = []

    for s in s_values:
        Pf, std_dev, beta = importance_sampling(L, lambda_vec, u_star, n_samples, s)
        is_results.append([s, Pf, std_dev, beta])
        print(f"s = {s} | Pf: {Pf:.5f} | Std Dev: {std_dev:.5f} | Beta: {beta:.4f}")

    # Convert results to DataFrame and save
    df_is_results = pd.DataFrame(is_results, columns=["s", "Pf", "Std Dev", "Beta"])
    df_is_results.to_csv("./results/is_sensitivity_analysis.csv", index=False)

    # Display the final table
    print(df_is_results)