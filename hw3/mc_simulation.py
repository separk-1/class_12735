import numpy as np
import scipy.stats as stats
import pandas as pd

# === Given mean (E[z]) and coefficient of variation (COV) ===
E_z = np.array([400, 300, 10, 1.4, 20])  
delta_z = np.array([0.15, 0.30, 0.20, 0.10, 0.20])  

# === Convert to lognormal distribution parameters ===
lambda_z = np.log(E_z / np.sqrt(1 + delta_z**2))
sigma_z = np.sqrt(np.log(1 + delta_z**2))

# === Given correlation matrix R ===
R = np.array([
    [1,  0.4,  0,  0,  0.4],
    [0.4, 1,  0,  0,  -0.2],
    [0,  0,  1,  0.3, 0.3],
    [0,  0,  0.3,  1,  -0.4],
    [0.4, -0.2,  0.3,  -0.4,  1]
])

# === Compute covariance matrix for lognormal space ===
cov_x = np.outer(sigma_z, sigma_z) * R

# === Cholesky decomposition for normal space transformation ===
L = np.linalg.cholesky(cov_x)

# === Limit state function g(z) ===
def limit_state_function(z):
    Fa, Fb, q, alpha, c = z
    F0 = 900  # Reference load
    d = q * ((Fa + Fb) / F0) ** alpha
    return np.log(c / d)

# === Q2.a: Crude Monte Carlo Simulation ===
def monte_carlo_simulation(n_samples):
    u_samples = np.random.randn(5, n_samples)  # Generate standard normal samples
    x_samples = L @ u_samples + lambda_z[:, None]  # Transform to normal space
    z_samples = np.exp(x_samples)  # Convert to physical space
    
    g_values = np.apply_along_axis(limit_state_function, 0, z_samples)
    failure_count = np.sum(g_values < 0)  # Count failures where g(z) < 0
    Pf = failure_count / n_samples  # Estimate failure probability
    
    beta = -stats.norm.ppf(Pf)  # Compute reliability index
    sigma_pf = np.sqrt(Pf * (1 - Pf) / n_samples)  # Compute standard deviation
    
    return Pf, sigma_pf, beta

# === Q2.b: Importance Sampling Simulation (s=1) ===
def importance_sampling(n_samples, u_star, s=1):
    u_samples = np.random.randn(5, n_samples) * s + u_star[:, None]  # Generate samples from importance distribution
    x_samples = L @ u_samples + lambda_z[:, None]  # Transform to normal space
    z_samples = np.exp(x_samples)  # Convert to physical space

    g_values = np.apply_along_axis(limit_state_function, 0, z_samples)
    weights = stats.multivariate_normal.pdf(u_samples.T, mean=np.zeros(5), cov=np.identity(5)) / \
              stats.multivariate_normal.pdf(u_samples.T, mean=u_star, cov=(s**2) * np.identity(5))  # Compute importance weights
    
    failure_count = np.sum(g_values < 0)
    Pf_IS = np.sum(weights[g_values < 0]) / n_samples  # Estimate failure probability using weighted samples
    
    beta_IS = -stats.norm.ppf(Pf_IS)  # Compute reliability index
    sigma_pf_IS = np.sqrt(np.var(weights[g_values < 0]) / n_samples)  # Compute standard deviation
    
    return Pf_IS, sigma_pf_IS, beta_IS

# === Perform Monte Carlo Simulation for various sample sizes ===
n_values = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]

mc_results = []
print("\nMonte Carlo Simulation Results:")
print(f"{'Samples':<10}{'Failure Prob':<20}{'Std Dev':<20}{'Reliability Index':<20}")
for n in n_values:
    Pf, sigma_pf, beta = monte_carlo_simulation(n)
    mc_results.append([n, Pf, sigma_pf, beta])
    print(f"{n:<10}{Pf:<20.6e}{sigma_pf:<20.6e}{beta:<20.6f}")

# Save Monte Carlo results to CSV
df_mc = pd.DataFrame(mc_results, columns=["Samples", "Failure Probability", "Std Dev", "Reliability Index"])
df_mc.to_csv("monte_carlo_results.csv", index=False)

# === Perform Importance Sampling (s=1) ===
u_star = np.array([0.092, 0.092, 0.022, -1.92, -0.89])  # Design point from FORM

is_results = []
print("\nImportance Sampling Results (s=1):")
print(f"{'Samples':<10}{'Failure Prob':<20}{'Std Dev':<20}{'Reliability Index':<20}")
for n in n_values:
    Pf_IS, sigma_pf_IS, beta_IS = importance_sampling(n, u_star, s=1)
    is_results.append([n, Pf_IS, sigma_pf_IS, beta_IS])
    print(f"{n:<10}{Pf_IS:<20.6e}{sigma_pf_IS:<20.6e}{beta_IS:<20.6f}")

# Save Importance Sampling results to CSV
df_is = pd.DataFrame(is_results, columns=["Samples", "Failure Probability", "Std Dev", "Reliability Index"])
df_is.to_csv("importance_sampling_results.csv", index=False)

# === Perform Importance Sampling for various s values (n=1000) ===
s_values = [0.1, 0.3, 1, 3, 10]
is_s_results = []
n = 1000
print("\nImportance Sampling Results for Different s values (n=1000):")
print(f"{'s Value':<10}{'Failure Prob':<20}{'Std Dev':<20}{'Reliability Index':<20}")
for s in s_values:
    Pf_IS, sigma_pf_IS, beta_IS = importance_sampling(n, u_star, s)
    is_s_results.append([s, Pf_IS, sigma_pf_IS, beta_IS])
    print(f"{s:<10}{Pf_IS:<20.6e}{sigma_pf_IS:<20.6e}{beta_IS:<20.6f}")

# Save Importance Sampling results for different s values
df_is_s = pd.DataFrame(is_s_results, columns=["s Value", "Failure Probability", "Std Dev", "Reliability Index"])
df_is_s.to_csv("importance_sampling_s_values.csv", index=False)

print("\nSimulation complete. Results saved to CSV files.")
