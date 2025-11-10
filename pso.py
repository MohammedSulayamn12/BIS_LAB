import numpy as np

# Objective function for bandwidth allocation
# Example: maximize fairness (sum of log of allocations)
def objective(allocation, demands, B_total):
    # Ensure allocations are within limits
    allocation = np.maximum(0, allocation)  # no negative bandwidth
    if np.sum(allocation) > B_total:  # normalize to fit within total bandwidth
        allocation = (allocation / np.sum(allocation)) * B_total
    
    # Utility: proportional fairness (logarithmic utility)
    utility = np.sum(np.log(1 + np.minimum(allocation, demands)))
    return utility

# Particle Swarm Optimization
def pso(num_users, demands, B_total, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    # Initialize particles
    particles = np.random.rand(num_particles, num_users) * (B_total / num_users)
    velocities = np.random.rand(num_particles, num_users) * 0.1
    
    # Initialize personal and global bests
    pbest = particles.copy()
    pbest_values = np.array([objective(p, demands, B_total) for p in particles])
    gbest = pbest[np.argmax(pbest_values)]
    gbest_value = np.max(pbest_values)
    
    for t in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(num_users), np.random.rand(num_users)
            velocities[i] = (w * velocities[i] 
                             + c1 * r1 * (pbest[i] - particles[i]) 
                             + c2 * r2 * (gbest - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Evaluate fitness
            fitness = objective(particles[i], demands, B_total)
            
            # Update personal best
            if fitness > pbest_values[i]:
                pbest[i] = particles[i].copy()
                pbest_values[i] = fitness
                
        # Update global best
        if np.max(pbest_values) > gbest_value:
            gbest = pbest[np.argmax(pbest_values)].copy()
            gbest_value = np.max(pbest_values)
            
        print(f"Iteration {t+1}, Best Utility = {gbest_value:.4f}")
    
    return gbest, gbest_value

# Example usage
if __name__ == "__main__":
    num_users = 5
    demands = np.array([10, 20, 15, 25, 30])  # user bandwidth demands
    B_total = 60  # total bandwidth available
    
    best_allocation, best_value = pso(num_users, demands, B_total)
    print("\nOptimal Bandwidth Allocation:", best_allocation)
    print("Maximized Utility:", best_value)
