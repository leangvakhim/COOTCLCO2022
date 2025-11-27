from wsn import wsn
from cootclco import cootclco
from benchmark import benchmark
import matplotlib.pyplot as plt
import numpy as np

print("1. Coverage")
print("2. Benchmark")
opt = int(input("Enter an options number: "))
if opt == 1:
    w, h = 100, 100
    sensor_nodes = 25
    rs = 10
    pop_size = 25
    max_iter = 500
    lb = 0
    ub = 100
    wsn_prob = wsn(w, h, sensor_nodes, rs, grid_res=2.0)
    dim = sensor_nodes * 2
    optimizer = cootclco(
        wsn_prob.objective_function,
        dim,
        pop_size,
        max_iter,
        lb,
        ub
    )

    best_score, best_pos, curve = optimizer.optimize()

    final_coverage = -best_score * 100
    best_sensors = best_pos.reshape((sensor_nodes, 2))
    wsn_prob.visualize_network(best_sensors, w, h, rs, final_coverage)
elif opt == 2:
    func_id = 2
    func, lb, ub, dim = benchmark.get_function(func_id)

    pop_size = 30
    max_iter = 500
    times = 30
    list_val = []
    for _ in range(times):
        optimizer = cootclco(
            func,
            dim,
            pop_size,
            max_iter,
            lb,
            ub
        )

        best_score, best_pos, curve = optimizer.optimize()
        list_val.append(best_score)

    mean_val = np.mean(list_val)
    std_val = np.std(list_val)
    min_val = np.max(list_val)
    max_val = np.min(list_val)

    print(f"Mean values: {mean_val:.4e}")
    print(f"Std values: {std_val:.4e}")
    print(f"Min values: {min_val:.4e}")
    print(f"Max values: {max_val:.4e}")

    # plt.figure(figsize=(10, 6))
    # plt.plot(curve, 'r-', linewidth=2, label=f'COOTCLCO (F{func_id})')
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Best Score (Log Scale)')
    # plt.title(f'Convergence Curve - Function F{func_id}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()