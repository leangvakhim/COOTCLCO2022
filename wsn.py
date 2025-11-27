import numpy as np
import matplotlib.pyplot as plt
from cootclco import cootclco

class wsn:
    def __init__(self, w, h, sensor_node, rs, grid_res):
        self.w = w
        self.h = h
        self.sensor_nodes = sensor_node
        self.rs = rs

        x_range = np.arange(0, w, grid_res)
        y_range = np.arange(0, h, grid_res)
        self.targets = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
        self.total_area_points = len(self.targets)

    def objective_function(self, position_vector):
        sensors = position_vector.reshape((self.sensor_nodes, 2))
        covered_count = 0
        for target in self.targets:
            distances = np.sqrt(np.sum((sensors - target) ** 2, axis=1))
            if np.any(distances <= self.rs):
                covered_count += 1

        coverage_rate = covered_count / self.total_area_points
        return -coverage_rate

    def visualize_network(sensors, w, h, rs, coverage):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')

        for i, (x, y) in enumerate(sensors):
            circle = plt.Circle((x, y), rs, color='blue', alpha=0.1)
            ax.add_patch(circle)
            ax.plot(x, y, 'r.', markersize=5)
            ax.text(x, y, str(i), fontsize=8)

        ax.set_title(f"WSN Coverages {coverage*100:.2f}")
        plt.grid(True)
        plt.show()
