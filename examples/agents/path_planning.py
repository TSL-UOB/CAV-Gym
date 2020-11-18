import copy
import csv
import math
from bisect import bisect
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot

from examples.agents.template import Agent
from library.geometry import Point

"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

TIMESTEPS = 500
ANIMATION_AREA = 20.0  # animation area length [m]
SHOW_ANIMATION = False


class Spline:  # Cubic spline class
    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = [xj - xi for xi, xj in zip(x, x[1:])]

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for ai, ci, hi, aj, cj in zip(self.a, self.c, h, self.a[1:], self.c[1:]):
            self.d.append((cj - ci) / (3.0 * hi))
            tb = (aj - ai) / hi - hi * (cj + 2.0 * ci) / 3.0
            self.b.append(tb)

    def calc(self, t):  # Calc position (if t is outside of the input x, return None)
        if t < self.x[0]:
            return None
        if t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def calcd(self, t):  # Calc first derivative (if t is outside of the input x, return None)
        if t < self.x[0]:
            return None
        if t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):  # Calc second derivative
        if t < self.x[0]:
            return None
        if t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):  # search data segment index
        return bisect(self.x, x) - 1

    def __calc_A(self, h):  # calc matrix A for spline coefficient c
        A = [[0.0 for _ in self.x] for _ in self.x]
        A[0][0] = 1.0
        for i, hi_hj in enumerate(zip(h, h[1:])):
            hi, hj = hi_hj
            if i != (self.nx - 2):
                A[i + 1][i + 1] = 2.0 * (hi + hj)
            A[i + 1][i] = hi
            A[i][i + 1] = hi

        A[0][1] = 0.0
        A[self.nx - 1][self.nx - 2] = 0.0
        A[self.nx - 1][self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):  # calc matrix B for spline coefficient c
        B = [0.0 for _ in self.x]
        for i, ai_hi_aj_hj_ak in enumerate(zip(self.a, h, self.a[1:], h[1:], self.a[2:])):
            ai, hi, aj, hj, ak = ai_hi_aj_hj_ak
            B[i + 1] = 3.0 * (ak - aj) / hj - 3.0 * (aj - ai) / hi
        return B


class Spline2D:  # 2D cubic spline class
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        total = 0
        cumulative_distances = [total]
        for xi, yi, xj, yj in zip(x, y, x[1:], y[1:]):
            distance = math.sqrt((xj - xi)**2 + (yj - yi)**2)
            total += distance
            cumulative_distances.append(total)
        return cumulative_distances

    def calc_position(self, s):  # calc position
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        return x, y

    def calc_curvature(self, s):  # calc curvature
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):  # calc yaw
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = [[time ** 3, time ** 4, time ** 5], [3 * time ** 2, 4 * time ** 3, 5 * time ** 4], [6 * time, 12 * time ** 2, 20 * time ** 3]]
        b = [xe - self.a0 - self.a1 * time - self.a2 * time ** 2, vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2]
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return xt


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = [[3 * time ** 2, 4 * time ** 3], [6 * time, 12 * time ** 2]]
        b = [vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2]
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt


class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


@dataclass(frozen=True)
class FrenetPlannerConstants:
    max_road_width: float  # maximum road width [m]
    d_road_w: float  # road width sampling length [m]
    min_t: float  # min prediction time [m]
    max_t: float  # max prediction time [m]
    dt: float  # time tick [s]
    target_speed: float  # target speed [m/s]
    d_t_s: float  # target speed sampling length [m/s]
    n_s_sample: float  # sampling number of target speed
    k_j: float  # cost weight
    k_t: float  # cost weight
    k_d: float  # cost weight
    k_lat: float  # cost weight
    k_lon: float  # cost weight
    robot_radius: float  # robot radius [m]
    max_speed: float  # maximum speed [m/s]
    max_accel: float  # maximum acceleration [m/ss]
    max_curvature: float  # maximum curvature [1/m]


class FrenetPlanner:
    def __init__(self, constants):
        self.constants = constants

    def calc_frenet_paths(self, frenet_state):
        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(-self.constants.max_road_width, self.constants.max_road_width, self.constants.d_road_w):

            # Lateral motion planning
            for Ti in np.arange(self.constants.min_t, self.constants.max_t, self.constants.dt):
                fp = FrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(frenet_state.c_d, frenet_state.c_d_d, frenet_state.c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, self.constants.dt)]
                fp.t = list(np.arange(0.0, Ti, self.constants.dt))
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(self.constants.target_speed - self.constants.d_t_s * self.constants.n_s_sample, self.constants.target_speed + self.constants.d_t_s * self.constants.n_s_sample, self.constants.d_t_s):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(frenet_state.s0, frenet_state.c_speed, 0.0, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum([element**2 for element in tfp.d_ddd])  # square of jerk
                    Js = sum([element**2 for element in tfp.s_ddd])  # square of jerk

                    # square of diff from target speed
                    ds = (self.constants.target_speed - tfp.s_d[-1]) ** 2

                    tfp.cd = self.constants.k_j * Jp + self.constants.k_t * Ti + self.constants.k_d * tfp.d[-1] ** 2
                    tfp.cv = self.constants.k_j * Js + self.constants.k_t * Ti + self.constants.k_d * ds
                    tfp.cf = self.constants.k_lat * tfp.cd + self.constants.k_lon * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def calc_global_paths(self, fplist, csp):
        for fp in fplist:

            # calc global positions
            for s, d in zip(fp.s, fp.d):
                x, y = csp.calc_position(s)
                if x is None:
                    break
                yaw = csp.calc_yaw(s)
                fx = x + d * math.cos(yaw + math.pi / 2.0)
                fy = y + d * math.sin(yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for xi, yi, xj, yj in zip(fp.x, fp.y, fp.x[1:], fp.y[1:]):
                dx = xj - xi
                dy = yj - yi
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.sqrt(dx**2 + dy**2))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for yaw, ds, next_yaw in zip(fp.yaw, fp.ds, fp.yaw[1:]):
                fp.c.append((next_yaw - yaw) / ds)

        return fplist

    def check_collision(self, fp, ob):
        for x, y in ob:
            d = [((ix - x) ** 2 + (iy - y) ** 2) for (ix, iy) in zip(fp.x, fp.y)]

            collision = any([di <= self.constants.robot_radius ** 2 for di in d])

            if collision:
                return False

        return True

    def check_paths(self, fplist, ob):
        ok_ind = []
        for fp in fplist:
            if any([v > self.constants.max_speed for v in fp.s_d]):  # Max speed check
                continue
            elif any([abs(a) > self.constants.max_accel for a in fp.s_dd]):  # Max accel check
                continue
            elif any([abs(c) > self.constants.max_curvature for c in fp.c]):  # Max curvature check
                continue
            elif not self.check_collision(fp, ob):
                continue

            ok_ind.append(fp)

        return ok_ind

    def frenet_optimal_planning(self, csp, frenet_state, ob):
        fplist = self.calc_frenet_paths(frenet_state)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist, ob)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return best_path

    def generate_target_course(self, x, y):
        csp = Spline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return rx, ry, ryaw, rk, csp


@dataclass(frozen=True)
class FrenetState:
    s0: float  # current course position
    c_d: float  # current lateral position [m]
    c_d_d: float  # current lateral speed [m/s]
    c_d_dd: float  # current lateral acceleration [m/s]
    c_speed: float  # current speed [m/s]

    def __iter__(self):
        yield self.s0
        yield self.c_d
        yield self.c_d_d
        yield self.c_d_dd
        yield self.c_speed


def read_test_data(path):
    data = []
    with open(path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            frenet_state = FrenetState(*[float(token) for token in row])
            data.append(frenet_state)
    return data


@dataclass(frozen=True)
class FrenetPoint:
    s: float  # longitudinal position (path position)
    d: float  # lateral position (left/right path offset)


def make_cartesian_point(frenet_point, spline2d):
    x, y = spline2d.calc_position(frenet_point.s)
    if x is None:
        return None
    yaw = spline2d.calc_yaw(frenet_point.s)
    fx = x + frenet_point.d * math.cos(yaw + math.pi / 2.0)
    fy = y + frenet_point.d * math.sin(yaw + math.pi / 2.0)
    return Point(x=fx, y=fy)


def main():
    test_data = read_test_data("frenet.csv")

    print(__file__ + " start!!")

    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # obstacle lists
    ob = [[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]]

    frenet_planner = FrenetPlanner(
        constants=FrenetPlannerConstants(
            max_road_width=7.0,  # maximum road width [m]
            d_road_w=1.0,  # road width sampling length [m]
            min_t=4.0,  # min prediction time [m]
            max_t=5.0,  # max prediction time [m]
            dt=0.2,  # time tick [s]
            target_speed=30.0 / 3.6,  # target speed [m/s]
            d_t_s=5.0 / 3.6,  # target speed sampling length [m/s]
            n_s_sample=1,  # sampling number of target speed
            k_j=0.1,  # cost weight
            k_t=0.1,  # cost weight
            k_d=1.0,  # cost weight
            k_lat=1.0,  # cost weight
            k_lon=1.0,  # cost weight
            robot_radius=2.0,  # robot radius [m]
            max_speed=50.0 / 3.6,  # maximum speed [m/s]
            max_accel=2.0,  # maximum acceleration [m/ss]
            max_curvature=1.0  # maximum curvature [1/m]
        )
    )

    tx, ty, tyaw, tc, csp = frenet_planner.generate_target_course(wx, wy)

    # initial state
    frenet_state = FrenetState(
        s0=0.0,  # current course position
        c_d=2.0,  # current lateral position [m]
        c_d_d=0.0,  # current lateral speed [m/s]
        c_d_dd=0.0,  # current lateral acceleration [m/s]
        c_speed=10.0 / 3.6  # current speed [m/s]
    )

    for i in range(TIMESTEPS):
        path = frenet_planner.frenet_optimal_planning(csp, frenet_state, ob)

        frenet_state = FrenetState(
            s0=path.s[1],
            c_d=path.d[1],
            c_d_d=path.d_d[1],
            c_d_dd=path.d_dd[1],
            c_speed=path.s_d[1]
        )
        print(make_cartesian_point(FrenetPoint(s=frenet_state.s0, d=frenet_state.c_d), csp))

        assert frenet_state == test_data[i], f"{frenet_state} == {test_data[i]}"

        if math.sqrt((path.x[1] - tx[-1])**2 + (path.y[1] - ty[-1])**2) <= 1.0:
            print("Goal")
            assert i == len(test_data) - 1
            break

        if SHOW_ANIMATION:  # pragma: no cover
            pyplot.cla()
            # for stopping simulation with the esc key.
            pyplot.gcf().canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])
            pyplot.plot(tx, ty)
            pyplot.plot([x for x, _ in ob], [y for _, y in ob], "xk")
            pyplot.plot(path.x[1:], path.y[1:], "-or")
            pyplot.plot(path.x[1], path.y[1], "vc")
            pyplot.xlim(path.x[1] - ANIMATION_AREA, path.x[1] + ANIMATION_AREA)
            pyplot.ylim(path.y[1] - ANIMATION_AREA, path.y[1] + ANIMATION_AREA)
            pyplot.title("v[km/h]:" + str(frenet_state.c_speed * 3.6)[0:4])
            pyplot.grid(True)
            pyplot.pause(0.0001)

    print("Finish")
    if SHOW_ANIMATION:  # pragma: no cover
        pyplot.grid(True)
        pyplot.pause(0.0001)
        pyplot.show()


if __name__ == "__main__":
    main()


class FrenetAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frenet_planner = FrenetPlanner(
            constants=FrenetPlannerConstants(
                max_road_width=7.0,  # maximum road width [m]
                d_road_w=1.0,  # road width sampling length [m]
                min_t=4.0,  # min prediction time [m]
                max_t=5.0,  # max prediction time [m]
                dt=0.2,  # time tick [s]
                target_speed=30.0 / 3.6,  # target speed [m/s]
                d_t_s=5.0 / 3.6,  # target speed sampling length [m/s]
                n_s_sample=1,  # sampling number of target speed
                k_j=0.1,  # cost weight
                k_t=0.1,  # cost weight
                k_d=1.0,  # cost weight
                k_lat=1.0,  # cost weight
                k_lon=1.0,  # cost weight
                robot_radius=2.0,  # robot radius [m]
                max_speed=50.0 / 3.6,  # maximum speed [m/s]
                max_accel=2.0,  # maximum acceleration [m/ss]
                max_curvature=1.0  # maximum curvature [1/m]
            )
        )

        self.wx = [0.0, 10.0, 20.5, 35.0, 70.5]
        self.wy = [0.0, -6.0, 5.0, 6.5, 0.0]
        self.tx, self.ty, self.tyaw, self.tc, self.csp = self.frenet_planner.generate_target_course(self.wx, self.wy)

    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        pass

    def process_feedback(self, previous_state, action, state, reward):
        pass
