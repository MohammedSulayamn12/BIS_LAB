

import math
import random
import numpy as np

def generate_users(n_users, width, height, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.uniform([0, 0], [width, height], size=(n_users, 2))

def generate_candidate_sites(grid_x, grid_y, width, height, radius, base_cost=1.0):
    xs = np.linspace(0, width, grid_x)
    ys = np.linspace(0, height, grid_y)
    sites = []
    for x in xs:
        for y in ys:
            cost = base_cost * (1.0 + 0.5 * random.random())
            sites.append({'pos': np.array([x, y]), 'radius': radius, 'cost': cost})
    return sites

def build_coverage_matrix(users, sites):
    n_users = users.shape[0]
    n_sites = len(sites)
    cov = np.zeros((n_sites, n_users), dtype=bool)
    for i, s in enumerate(sites):
        d2 = np.sum((users - s['pos'])**2, axis=1)
        cov[i, :] = d2 <= (s['radius']**2)
    return cov

def solution_metrics(selected_indices, cov_matrix, sites):
    if len(selected_indices) == 0:
        return 0.0, 0.0
    covered = np.any(cov_matrix[selected_indices, :], axis=0)
    coverage_fraction = covered.mean()
    total_cost = sum(sites[i]['cost'] for i in selected_indices)
    return coverage_fraction, total_cost

def aco_bs_placement(users, sites, cov_matrix,
                     n_ants=30, n_iters=200,
                     alpha=1.0, beta=3.0, rho=0.1, q=1.0,
                     max_sites=None, coverage_weight=0.7, cost_weight=0.3,
                     seed=None):
    rnd = np.random.RandomState(seed)
    n_sites = len(sites)
    if max_sites is None:
        max_sites = max(1, n_sites // 10)

    base_gain = np.maximum(cov_matrix.sum(axis=1).astype(float), 1.0)
    eta = base_gain / np.array([s['cost'] for s in sites]) 
    tau = np.ones(n_sites) 

    max_cost_possible = sum(s['cost'] for s in sites)

    best = {'indices': [], 'coverage': 0.0, 'cost': 0.0, 'fitness': float('inf')}

    for iter_idx in range(n_iters):
        ants_solutions = []
        for ant in range(n_ants):
            selected = []
            covered_mask = np.zeros(users.shape[0], dtype=bool)
            available = set(range(n_sites))

            for step in range(max_sites):
                gains = []
                for i in available:
                    new_covered = np.sum(np.logical_and(~covered_mask, cov_matrix[i]))
                    gains.append((i, (new_covered + 1e-6) / sites[i]['cost']))
                idxs, gains_vals = zip(*gains)
                tau_vals = np.array([tau[i] for i in idxs])**alpha
                eta_vals = np.array(gains_vals)**beta
                probs = tau_vals * eta_vals
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    break
                probs = probs / probs_sum
                choice = rnd.choice(len(idxs), p=probs)
                chosen_site = idxs[choice]
                selected.append(chosen_site)
                covered_mask = np.logical_or(covered_mask, cov_matrix[chosen_site])
                available.remove(chosen_site)

                if covered_mask.mean() > 0.999:
                    break

            coverage, total_cost = solution_metrics(selected, cov_matrix, sites)
            fitness = coverage_weight * (1.0 - coverage) + cost_weight * (total_cost / max_cost_possible)
            ants_solutions.append({'indices': selected, 'coverage': coverage, 'cost': total_cost, 'fitness': fitness})


        ants_solutions.sort(key=lambda x: x['fitness'])
        best_ant = ants_solutions[0]
 
        if best_ant['fitness'] < best['fitness']:
            best = best_ant.copy()

 
        tau *= (1.0 - rho)
 
        n_elite = max(1, n_ants // 10)
        for k in range(n_elite):
            sol = ants_solutions[k]
            deposit = q / (1e-6 + sol['fitness'])
            for i in sol['indices']:
                tau[i] += deposit

        tau = np.clip(tau, 1e-6, 1e6)

    return best, tau

def example_run():
    width, height = 100, 100
    n_users = 1000
    users = generate_users(n_users, width, height, seed=42)

    grid_x, grid_y = 10, 10
    radius = 18.0
    sites = generate_candidate_sites(grid_x, grid_y, width, height, radius, base_cost=5.0)
    cov_matrix = build_coverage_matrix(users, sites)

    best, pheromones = aco_bs_placement(users, sites, cov_matrix,
                                       n_ants=40, n_iters=200,
                                       alpha=1.0, beta=4.0, rho=0.2, q=1.0,
                                       max_sites=8,
                                       coverage_weight=0.7, cost_weight=0.3,
                                       seed=1)

    print("Best solution found:")
    print(f"  Selected sites: {sorted(best['indices'])}")
    print(f"  Coverage: {best['coverage']*100:.2f}%")
    print(f"  Total cost: {best['cost']:.2f}")
    print(f"  Fitness: {best['fitness']:.6f}")

if __name__ == '__main__':
    example_run()



output:
Best solution found:
  Selected sites: [11, 17, 24, 48, 55, 62, 77, 84]
  Coverage: 75.90%
  Total cost: 46.69
  Fitness: 0.191186
