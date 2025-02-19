import numpy as np
import scipy.stats as stats

def compute_next_u(L, lambda_vec, u_k):
    # Calculate x_k and its exponential values
    x_k = np.dot(L, u_k) + lambda_vec
    Fa, Fb, q, alpha, c = np.exp(x_k)
    Fa, Fb, q, alpha, c = Fa.item(), Fb.item(), q.item(), alpha.item(), c.item()
    
    # Calculate gz_gradient
    gz_grad = np.array([
        [-alpha/(Fa + Fb)],
        [-alpha/(Fa + Fb)],
        [-1/q],
        [-np.log((Fa + Fb)/900)],
        [1/c]
    ])
    
    # Calculate limited space function
    gu_u = x_k[4] - x_k[2] - np.exp(x_k[3]) * np.log((np.exp(x_k[0]) + np.exp(x_k[1]))/900)
    
    # Calculate gradient
    J_zx = np.diag(np.exp(x_k).T.squeeze(0))
    gx_grad = np.dot(J_zx.T, gz_grad)
    gu_grad = np.dot(L.T, gx_grad)
    
    # Calculate next u
    numerator = (np.dot(gu_grad.T, u_k) - gu_u) * gu_grad
    denominator = np.dot(gu_grad.T, gu_grad)
    u_k_next = numerator / denominator
    
    return u_k_next, gu_grad, gu_u

def main():
    # Initialize parameters
    L = np.array([
        [0.149, 0, 0, 0, 0],
        [0.117, 0.269, 0, 0, 0],
        [0, 0, 0.198, 0, 0],
        [0, 0, 0.029, 0.096, 0],
        [0.079, -0.078, 0.060, -0.101, 0.115]
    ])
    lambda_vec = np.array([[5.980], [5.661], [2.283], [0.331], [2.976]])
    u_k = np.zeros((5, 1))
    
    # Iterate 5 times
    for i in range(5):
        u_k_next, gu_grad, gu_u = compute_next_u(L, lambda_vec, u_k)
        u_k = u_k_next
        
        # Calculate reliability index and failure probability
        rel_index = np.linalg.norm(u_k_next.T.squeeze(0))
        failure_probability = stats.norm.cdf(-rel_index)
        
        # Print results
        print(f"\nIteration {i+1}")
        print(f"Updated u: {u_k_next.T.squeeze(0)}")
        print(f"Gradient u: {gu_grad.T.squeeze(0)}")
        print(f"Limited space function value: {gu_u.item()}")
        print(f"Reliability index: {rel_index}")
        print(f"Failure probability: {failure_probability}")

if __name__ == "__main__":
    main()
