import numpy as np
import math
import matplotlib.pyplot as plt
import random
from frenet_optimal_trajectory import generate_target_course, frenet_optimal_planning


# way points -- AG_comment: This is to be replaced with the fixed waypoints for the AV path. 
wx = [0.0, 10.0, 20.5, 35.0, 70.5]
wy = [0.0, -6.0, 5.0, 6.5, 0.0]

# csp  - is cubic spline planner, the variables below are generate from the csp. Check out generate_target_course for further info.
# tx   - is x coordinates for path
# ty   - is y coordinates for path 
# tyaw - is yaw values for points in path
# tc   - is curvature for different points in path
tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)


# initial state
c_speed = 10.0 / 3.6  # current speed [m/s]
c_d = 2.0  # current lateral position [m]
c_d_d = 0.0  # current lateral speed [m/s]
c_d_dd = 0.0  # current lateral acceleration [m/s]
s0 = 0.0  # current course position

area = 40.0  # animation area length [m]



SIM_LOOP = 500
show_animation = True
for i in range(SIM_LOOP):

    # ob - obstacle lists -- AG_comment: This is to be continiously updated with the position of the dynamic agents
    n = random.uniform(-1, 1)*5
    ob = np.array([[20.0+n, 10.0+n],
               [30.0, 6.0],
               [30.0+n, 8.0+n],
               [35.0, 8.0],
               [50.0, 3.0]
               ])



    path = frenet_optimal_planning(
        csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

    s0 = path.s[1]
    c_d = path.d[1]
    c_d_d = path.d_d[1]
    c_d_dd = path.d_dd[1]
    c_speed = path.s_d[1]

    if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
        print("Goal")
        break

    if show_animation:  # pragma: no cover
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(tx, ty)
        plt.plot(ob[:, 0], ob[:, 1], "xk")
        plt.plot(path.x[1:], path.y[1:], "-or")
        plt.plot(path.x[1], path.y[1], "vc")
        plt.xlim(path.x[1] - area, path.x[1] + area)
        plt.ylim(path.y[1] - area, path.y[1] + area)
        plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
        plt.grid(True)
        plt.pause(0.0001)

print("Finish")
if show_animation:  # pragma: no cover
    plt.grid(True)
    plt.pause(0.0001)
    plt.show()

