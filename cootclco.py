import numpy as np
from tqdm import tqdm
import math

class cootclco:
    def __init__(self, obj_func, dim, pop_size, max_iter, lb, ub, leader_node=None):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.array(lb) if isinstance(lb, list) else lb
        self.ub = np.array(ub) if isinstance(ub, list) else ub

        # number of leaders and followers
        if leader_node is None:
            self.leader_node = max(1, int(0.1 * pop_size))
        else:
            self.leader_node = leader_node
        self.follower_node = pop_size - self.leader_node

        self.gBest_score = float('inf')
        self.gBest_pos = np.zeros(dim)

        self.convergence_curve = []

    def chaotic_tent_map_initialization(self):
        positions = np.zeros((self.pop_size, self.dim))

        for j in range(self.dim):
            z = np.random.rand()
            for i in range(self.pop_size):
                # eq 18
                if z < 0.5:
                    z = 2 * z
                else:
                    z = 2 * (1 - z)
                # eq 17
                positions[i, j] = self.lb + z * (self.ub - self.lb)

        return positions

    def levy_flight(self, beta=1.5):
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, sigma_v, self.dim)

        step = u / (np.abs(v) ** (1 / beta))
        return step

    def optimize(self):
        pop_pos = self.chaotic_tent_map_initialization()
        pop_fit = np.array([self.obj_func(ind) for ind in pop_pos])

        sorted_indices = np.argsort(pop_fit)
        leaders_pos = pop_pos[sorted_indices[:self.leader_node]].copy()
        leaders_fit = pop_fit[sorted_indices[:self.leader_node]].copy()
        followers_pos = pop_pos[sorted_indices[self.leader_node:]].copy()
        followers_fit = pop_fit[sorted_indices[self.leader_node:]].copy()

        self.gBest_score = leaders_fit[0]
        self.gBest_pos = leaders_pos[0].copy()

        for iter_count in tqdm(range(self.max_iter), desc="COOTCLCO Progress: "):
            # eq 11 & 16
            a = 1 - iter_count * (1 / self.max_iter)
            b = 2 - iter_count * (1 / self.max_iter)

            for i in range(self.follower_node):
                k = (i % self.leader_node)
                selected_leader = leaders_pos[k]

                r1 = np.random.rand()
                r2 = np.random.rand()
                r = 2 * np.random.rand() - 1

                levy_step = self.levy_flight()

                if np.random.rand() < 0.5:
                    # eq 26
                    q = np.random.uniform(self.lb, self.ub, self.dim)
                    followers_pos[i] = followers_pos[i] + a * r2 * (q - followers_pos[i]) * levy_step
                else:
                    # eq 27
                    if i > 0:
                        followers_pos[i] = (followers_pos[i-1] + followers_pos[i]) / 2 * levy_step
                    else:
                        followers_pos[i] = followers_pos[i]

                followers_pos[i] = selected_leader + 2 * r1 * math.cos(2 * math.pi * r) * (selected_leader - followers_pos[i]) * levy_step
                followers_pos[i] = np.clip(followers_pos[i], self.lb, self.ub)
                fitness = self.obj_func(followers_pos[i])
                followers_fit[i] = fitness

                if fitness < self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_pos = followers_pos[i].copy()

            for i in range(self.leader_node):
                r3 = np.random.rand()
                r4 = np.random.rand()
                r = 2 * np.random.rand() - 1

                # eq 15
                if r4 < 0.5:
                    leaders_pos[i] = b * r3 * math.cos(2 * math.pi * r) * (self.gBest_pos - leaders_pos[i]) + self.gBest_pos
                else:
                    leaders_pos[i] = b * r3 * math.cos(2 * math.pi * r) * (self.gBest_pos - leaders_pos[i]) - self.gBest_pos

                leaders_pos[i] = np.clip(leaders_pos[i], self.lb, self.ub)

                fitness = self.obj_func(leaders_pos[i])
                leaders_fit[i] = fitness

                if fitness < self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_pos = leaders_pos[i].copy()
            # eq 39
            ps = np.exp(-(iter_count / self.max_iter) ** 20) + 0.05

            current_best_pos = self.gBest_pos.copy()
            new_pos = np.zeros_like(current_best_pos)

            if np.random.rand() < ps:
                # eq 32
                r = np.random.rand(self.dim)
                x_prime = self.lb + self.ub - current_best_pos

                # eq 33
                b1 = (1 - iter_count / self.max_iter) ** iter_count
                r1 = np.random.rand()
                r = 2 * np.random.rand() - 1
                new_pos = x_prime + b1 * 2 * r1 * math.cos(2*math.pi*r) * (x_prime - current_best_pos)
            else:
                # eq 38
                cauchy_val = np.random.standard_cauchy(self.dim)
                new_pos = current_best_pos + cauchy_val * current_best_pos

            new_pos = np.clip(new_pos, self.lb, self.ub)
            new_fit = self.obj_func(new_pos)

            if new_fit < self.gBest_score:
                self.gBest_score = new_fit
                self.gBest_pos = new_pos.copy()

            self.convergence_curve.append(self.gBest_score)

        return self.gBest_score, self.gBest_pos, self.convergence_curve
