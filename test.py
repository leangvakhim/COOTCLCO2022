from wsn import wsn
from cootclco import cootclco

print("1. Coverage")
print("2. Benchmark")
opt = int(input("Enter an options number: "))
if opt == 1:
    w, h = 100, 100
    sensor_nodes = 25
    rs = 10
    pop_size = 30
    max_iter = 100
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
    a =0