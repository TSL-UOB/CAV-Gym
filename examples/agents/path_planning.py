import copy
import csv
import math
from bisect import bisect
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot

from examples.agents.template import Agent
from library.geometry import Point
from reporting import pretty_float

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
                frenet_path = FrenetPath()

                lat_qp = QuinticPolynomial(frenet_state.d, frenet_state.d_d, frenet_state.d_dd, di, 0.0, 0.0, Ti)

                frenet_path.t = list(np.arange(0.0, Ti, self.constants.dt))
                frenet_path.d = [lat_qp.calc_point(t) for t in frenet_path.t]  # lateral position [m]
                frenet_path.d_d = [lat_qp.calc_first_derivative(t) for t in frenet_path.t]  # lateral speed [m/s]
                frenet_path.d_dd = [lat_qp.calc_second_derivative(t) for t in frenet_path.t]  # lateral acceleration [m/s]
                frenet_path.d_ddd = [lat_qp.calc_third_derivative(t) for t in frenet_path.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(self.constants.target_speed - self.constants.d_t_s * self.constants.n_s_sample, self.constants.target_speed + self.constants.d_t_s * self.constants.n_s_sample, self.constants.d_t_s):
                    frenet_path_prime = copy.deepcopy(frenet_path)
                    lon_qp = QuarticPolynomial(frenet_state.s, frenet_state.s_d, 0.0, tv, 0.0, Ti)

                    frenet_path_prime.s = [lon_qp.calc_point(t) for t in frenet_path.t]  # longitudinal position
                    frenet_path_prime.s_d = [lon_qp.calc_first_derivative(t) for t in frenet_path.t]  # longitudinal speed
                    frenet_path_prime.s_dd = [lon_qp.calc_second_derivative(t) for t in frenet_path.t]
                    frenet_path_prime.s_ddd = [lon_qp.calc_third_derivative(t) for t in frenet_path.t]

                    Jp = sum([element**2 for element in frenet_path_prime.d_ddd])  # square of jerk
                    Js = sum([element**2 for element in frenet_path_prime.s_ddd])  # square of jerk

                    # square of diff from target speed
                    ds = (self.constants.target_speed - frenet_path_prime.s_d[-1]) ** 2

                    frenet_path_prime.cd = self.constants.k_j * Jp + self.constants.k_t * Ti + self.constants.k_d * frenet_path_prime.d[-1] ** 2
                    frenet_path_prime.cv = self.constants.k_j * Js + self.constants.k_t * Ti + self.constants.k_d * ds
                    frenet_path_prime.cf = self.constants.k_lat * frenet_path_prime.cd + self.constants.k_lon * frenet_path_prime.cv

                    frenet_paths.append(frenet_path_prime)

        return frenet_paths

    def calc_global_paths(self, frenet_paths, spline2d):
        for frenet_path in frenet_paths:

            # calc global positions
            for s, d in zip(frenet_path.s, frenet_path.d):
                x, y = spline2d.calc_position(s)
                if x is None:
                    break
                yaw = spline2d.calc_yaw(s)
                fx = x + d * math.cos(yaw + math.pi / 2.0)
                fy = y + d * math.sin(yaw + math.pi / 2.0)
                frenet_path.x.append(fx)
                frenet_path.y.append(fy)

            # calc yaw and ds
            for xi, yi, xj, yj in zip(frenet_path.x, frenet_path.y, frenet_path.x[1:], frenet_path.y[1:]):
                dx = xj - xi
                dy = yj - yi
                frenet_path.yaw.append(math.atan2(dy, dx))
                frenet_path.ds.append(math.sqrt(dx**2 + dy**2))

            frenet_path.yaw.append(frenet_path.yaw[-1])
            frenet_path.ds.append(frenet_path.ds[-1])

            # calc curvature
            for yaw, ds, next_yaw in zip(frenet_path.yaw, frenet_path.ds, frenet_path.yaw[1:]):
                frenet_path.c.append((next_yaw - yaw) / ds)

        return frenet_paths

    def check_collision(self, frenet_path, obstacles):
        for x, y in obstacles:
            d = [math.sqrt((ix - x) ** 2 + (iy - y) ** 2) for (ix, iy) in zip(frenet_path.x, frenet_path.y)]

            collision = any([di <= self.constants.robot_radius for di in d])

            if collision:
                return False

        return True

    def check_paths(self, frenet_paths, obstacles):
        ok_ind = []
        for frenet_path in frenet_paths:
            if any([v > self.constants.max_speed for v in frenet_path.s_d]):  # Max speed check
                continue
            elif any([abs(a) > self.constants.max_accel for a in frenet_path.s_dd]):  # Max accel check
                continue
            elif any([abs(c) > self.constants.max_curvature for c in frenet_path.c]):  # Max curvature check
                continue
            elif not self.check_collision(frenet_path, obstacles):
                continue

            ok_ind.append(frenet_path)

        return ok_ind

    def frenet_optimal_planning(self, spline2d, frenet_state, obstacles):
        frenet_paths = self.calc_frenet_paths(frenet_state)
        frenet_paths = self.calc_global_paths(frenet_paths, spline2d)
        frenet_paths = self.check_paths(frenet_paths, obstacles)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for frenet_path in frenet_paths:
            if min_cost >= frenet_path.cf:
                min_cost = frenet_path.cf
                best_path = frenet_path

        return best_path

    def generate_target_course(self, x, y):
        spline2d = Spline2D(x, y)
        s = np.arange(0, spline2d.s[-1], 0.1)

        targets = []
        for si in s:
            xi, yi = spline2d.calc_position(si)
            target = Target(
                position=Point(x=xi, y=yi),
                yaw=spline2d.calc_yaw(si),
                curvature=spline2d.calc_curvature(si)
            )
            targets.append(target)

        return targets, spline2d


@dataclass(frozen=True)
class FrenetState:
    s: float  # longitudinal position
    s_d: float  # longitudinal speed [m/s]
    s_dd: float
    s_ddd: float
    d: float  # lateral position [m]
    d_d: float  # lateral speed [m/s]
    d_dd: float  # lateral acceleration [m/s]
    d_ddd: float

    def __iter__(self):
        yield self.s
        yield self.s_d
        yield self.s_dd
        yield self.s_ddd
        yield self.d
        yield self.d_d
        yield self.d_dd
        yield self.d_ddd


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


@dataclass(frozen=True)
class Target:
    position: Point
    yaw: float  # yaw
    curvature: float  # curvature


def make_cartesian_point(frenet_point, spline2d):
    x, y = spline2d.calc_position(frenet_point.s)
    if x is None or y is None:
        return None
    yaw = spline2d.calc_yaw(frenet_point.s)
    fx = x + frenet_point.d * math.cos(yaw + math.pi / 2.0)
    fy = y + frenet_point.d * math.sin(yaw + math.pi / 2.0)
    return Point(x=fx, y=fy)


def calc_distance(left_point, right_point):
    return math.sqrt((right_point.x - left_point.x)**2 + (right_point.y - left_point.y)**2)


def find_closest_spline_point_index(point, spline_points):
    closest_distance = math.inf
    closest_spline_point_index = 1

    for i, reference in enumerate(spline_points):
        distance = calc_distance(point, reference)
        if distance < closest_distance:
            closest_distance = distance
            closest_spline_point_index = i

    if closest_spline_point_index >= len(spline_points) - 1:
        closest_2nd_spline_point_index = closest_spline_point_index - 1
    elif closest_spline_point_index == 0:
        closest_2nd_spline_point_index = closest_spline_point_index + 1
    else:
        reference_p1 = spline_points[closest_spline_point_index + 1]
        distance_p1 = calc_distance(point, reference_p1)

        reference_m1 = spline_points[closest_spline_point_index - 1]
        distance_m1 = calc_distance(point, reference_m1)

        if distance_m1 < distance_p1:
            closest_2nd_spline_point_index = closest_spline_point_index - 1
        else:
            closest_2nd_spline_point_index = closest_spline_point_index + 1

    return closest_spline_point_index, closest_2nd_spline_point_index


def make_frenet_point(point, spline_points):
    closest_spline_point_index, closest_2nd_spline_point_index = find_closest_spline_point_index(point, spline_points)

    if closest_spline_point_index > closest_2nd_spline_point_index:
        next_spline_point_index = closest_spline_point_index
    else:
        next_spline_point_index = closest_2nd_spline_point_index

    previous_spline_point_index = next_spline_point_index - 1
    if next_spline_point_index == 0:
        previous_spline_point_index = 0
        next_spline_point_index = 1

    previous_spline_point = spline_points[previous_spline_point_index]
    next_spline_point = spline_points[next_spline_point_index]

    tangent_x = next_spline_point.x - previous_spline_point.x
    tangent_y = next_spline_point.y - previous_spline_point.y

    vector_x = point.x - previous_spline_point.x
    vector_y = point.y - previous_spline_point.y

    tangent_length = math.sqrt(tangent_x**2 + tangent_y**2)
    projected_vector_norm = (vector_x * tangent_x + vector_y * tangent_y) / tangent_length
    projected_vector_x = projected_vector_norm * tangent_x / tangent_length
    projected_vector_y = projected_vector_norm * tangent_y / tangent_length

    frenet_d = calc_distance(Point(x=vector_x, y=vector_y), Point(x=projected_vector_x, y=projected_vector_y))

    d = (point.x - previous_spline_point.x) * (next_spline_point.y - previous_spline_point.y) - (point.y - previous_spline_point.y) * (next_spline_point.x - previous_spline_point.x)
    side = 1 if d > 0 else -1
    if side > 0:
        frenet_d = frenet_d * -1

    frenet_s = 0
    for i in range(previous_spline_point_index):
        frenet_s = frenet_s + calc_distance(spline_points[i], spline_points[i + 1])

    frenet_s = frenet_s + projected_vector_norm

    return FrenetPoint(s=frenet_s, d=frenet_d)


def my_make_frenet_point(point, spline_points):
    spline_distances = [calc_distance(point, reference) for reference in spline_points]

    closest_mean_distance = math.inf
    closest_pair = None
    closest_index = None
    for i, element in enumerate(zip(spline_points, spline_distances, spline_points[1:], spline_distances[1:])):
        spline_point_first, distance_first, spline_point_second, distance_second = element
        mean_distance = (distance_first + distance_second) / 2
        if mean_distance < closest_mean_distance:
            closest_mean_distance = mean_distance
            closest_pair = spline_point_first, spline_point_second
            closest_index = i

    preceding_spline_points = spline_points[:closest_index]  # select spline points up to but excluding closest_pair
    spline_point_first, spline_point_second = closest_pair

    tangent_x = spline_point_second.x - spline_point_first.x
    tangent_y = spline_point_second.y - spline_point_first.y

    vector_x = point.x - spline_point_first.x
    vector_y = point.y - spline_point_first.y

    tangent_length = math.sqrt(tangent_x**2 + tangent_y**2)
    projected_vector_norm = (vector_x * tangent_x + vector_y * tangent_y) / tangent_length

    projected_vector_x = projected_vector_norm * tangent_x / tangent_length
    projected_vector_y = projected_vector_norm * tangent_y / tangent_length

    frenet_d = calc_distance(Point(x=vector_x, y=vector_y), Point(x=projected_vector_x, y=projected_vector_y))

    d = (point.x - spline_point_first.x) * (spline_point_second.y - spline_point_first.y) - (point.y - spline_point_first.y) * (spline_point_second.x - spline_point_first.x)
    frenet_d = math.copysign(frenet_d, -d)  # sign of frenet_d should be opposite of d

    frenet_s = projected_vector_norm
    for preceding_spline_point, next_preceding_spline_point in zip(preceding_spline_points, preceding_spline_points[1:]):
        frenet_s += calc_distance(preceding_spline_point, next_preceding_spline_point)

    return FrenetPoint(s=frenet_s, d=frenet_d)


def main():
    test_data = read_test_data("frenet.csv")

    print(__file__ + " start!!")

    waypoints = [Point(x=0.0, y=0.0), Point(x=10.0, y=-6.0), Point(x=20.5, y=5.0), Point(x=35.0, y=6.5), Point(x=70.5, y=0.0)]
    obstacles = [Point(x=20.0, y=10.0), Point(x=30.0, y=6.0), Point(x=30.0, y=8.0), Point(x=35.0, y=8.0), Point(x=50.0, y=3.0)]

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

    targets, spline2d = frenet_planner.generate_target_course([waypoint.x for waypoint in waypoints], [waypoint.y for waypoint in waypoints])

    # initial state
    frenet_state = FrenetState(
        s=0.0,  # initial longitudinal position
        s_d=10.0 / 3.6,  # initial longitudinal speed [m/s]
        s_dd=0.0,
        s_ddd=0.0,
        d=2.0,  # initial lateral position [m]
        d_d=0.0,  # initial lateral speed [m/s]
        d_dd=0.0,  # initial lateral acceleration [m/s]
        d_ddd=0.0
    )

    comparisons = 0
    draws_s = 0
    matlab_wins_s = 0
    my_wins_s = 0
    draws_d = 0
    matlab_wins_d = 0
    my_wins_d = 0
    draws = 0
    matlab_wins = 0
    my_wins = 0
    matlab_total_error_s = 0
    matlab_total_error_d = 0
    matlab_total_error = 0
    my_total_error_s = 0
    my_total_error_d = 0
    my_total_error = 0
    for i in range(TIMESTEPS):
        path = frenet_planner.frenet_optimal_planning(spline2d, frenet_state, obstacles)

        frenet_state = FrenetState(
            s=path.s[1],  # initial longitudinal position
            s_d=path.s_d[1],  # initial longitudinal speed [m/s]
            s_dd=path.s_dd[1],
            s_ddd=path.s_ddd[1],
            d=path.d[1],  # initial lateral position [m]
            d_d=path.d_d[1],  # initial lateral speed [m/s]
            d_dd=path.d_dd[1],  # initial lateral acceleration [m/s]
            d_ddd=path.d_ddd[1]
        )

        target_points = [target.position for target in targets]
        path_points = [Point(x=x, y=y) for x, y in zip(path.x, path.y)]

        def calc_errors(left_frenet_point, right_frenet_point):
            distance_s = right_frenet_point.s - left_frenet_point.s
            distance_d = right_frenet_point.d - left_frenet_point.d
            distance = math.sqrt(distance_s**2 + distance_d**2)
            return abs(distance_s), abs(distance_d), abs(distance)

        frenet_path_points = [FrenetPoint(s=s, d=d) for s, d in zip(path.s, path.d)]
        for frenet_point in frenet_path_points:
            cartesian_point = make_cartesian_point(frenet_point, spline2d)
            if cartesian_point is not None:
                matlab_frenet_point = make_frenet_point(cartesian_point, target_points)
                my_frenet_point = my_make_frenet_point(cartesian_point, target_points)
                matlab_error_s, matlab_error_d, matlab_error = calc_errors(frenet_point, matlab_frenet_point)
                my_error_s, my_error_d, my_error = calc_errors(frenet_point, my_frenet_point)
                if matlab_error_s < my_error_s:
                    matlab_wins_s += 1
                elif matlab_error_s > my_error_s:
                    my_wins_s += 1
                else:
                    draws_s += 1
                if matlab_error_d < my_error_d:
                    matlab_wins_d += 1
                elif matlab_error_d > my_error_d:
                    my_wins_d += 1
                else:
                    draws_d += 1
                if matlab_error < my_error:
                    matlab_wins += 1
                elif matlab_error > my_error:
                    my_wins += 1
                else:
                    draws += 1
                comparisons += 1
                matlab_total_error_s += matlab_error_s
                matlab_total_error_d += matlab_error_d
                matlab_total_error += matlab_error
                my_total_error_s += my_error_s
                my_total_error_d += my_error_d
                my_total_error += my_error

        translated_path_points = [make_cartesian_point(my_make_frenet_point(point, target_points), spline2d) for point in path_points]
        translated_path_points = [point for point in translated_path_points if point is not None]
        translated_obstacles = [make_cartesian_point(my_make_frenet_point(obstacle, target_points), spline2d) for obstacle in obstacles]
        translated_obstacles = [point for point in translated_obstacles if point is not None]

        assert frenet_state == test_data[i], f"{frenet_state} == {test_data[i]}"

        if math.sqrt((path.x[1] - targets[-1].position.x)**2 + (path.y[1] - targets[-1].position.y)**2) <= 1.0:  # ig distance from current position to last target position is less than error threshold
            print("Goal")
            assert i == len(test_data) - 1
            break

        if SHOW_ANIMATION:  # pragma: no cover
            pyplot.cla()
            pyplot.gca().set_aspect("equal", adjustable="box")
            # for stopping simulation with the esc key.
            pyplot.gcf().canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

            pyplot.plot([target.position.x for target in targets], [target.position.y for target in targets], color="blue", label="target spline")
            pyplot.plot(path.x[1:], path.y[1:], marker="x", color="green", label="plan")
            pyplot.plot([point.x for point in translated_path_points[1:]], [point.y for point in translated_path_points[1:]], marker="x", color="orange", label="plan (translated)")

            pyplot.scatter(path.x[1], path.y[1], color="red", label="ego")
            pyplot.scatter([obstacle.x for obstacle in obstacles], [obstacle.y for obstacle in obstacles], marker="x", color="purple", label="obstacles")
            pyplot.scatter([obstacle.x for obstacle in translated_obstacles], [obstacle.y for obstacle in translated_obstacles], marker="x", color="orange", label="obstacles (translated)")

            pyplot.gca().add_artist(pyplot.Circle((path.x[1], path.y[1]), radius=frenet_planner.constants.robot_radius, color="red"))

            pyplot.xlim(path.x[1] - ANIMATION_AREA * 0.25, path.x[1] + ANIMATION_AREA * 1.75)
            pyplot.ylim(path.y[1] - ANIMATION_AREA, path.y[1] + ANIMATION_AREA)
            pyplot.title(f"t={i}, v[km/h]={frenet_state.s_d * 3.6}")
            pyplot.legend(loc="lower left")
            pyplot.grid(True)
            pyplot.tight_layout()
            pyplot.pause(0.0001)

    print("Finish")
    if SHOW_ANIMATION:  # pragma: no cover
        pyplot.grid(True)
        pyplot.pause(0.0001)
        pyplot.show()

    assert comparisons == draws_s + matlab_wins_s + my_wins_s
    assert comparisons == draws_d + matlab_wins_d + my_wins_d
    assert comparisons == draws + matlab_wins + my_wins
    print(f"{comparisons} error comparisons\n"
          f"- {draws} draws ({draws / comparisons:.0%})\n"
          f"- {matlab_wins} matlab wins ({matlab_wins / comparisons:.0%})\n"
          f"- {my_wins} my wins ({my_wins / comparisons:.0%})\n"
          f"\n"
          f"{comparisons} s-error comparisons\n"
          f"- {draws_s} draws ({draws_s / comparisons:.0%})\n"
          f"- {matlab_wins_s} matlab wins ({matlab_wins_s / comparisons:.0%})\n"
          f"- {my_wins_s} my wins ({my_wins_s / comparisons:.0%})\n"
          f"\n"
          f"{comparisons} d-error comparisons\n"
          f"- {draws_d} draws ({draws_d / comparisons:.0%})\n"
          f"- {matlab_wins_d} matlab wins ({matlab_wins_d / comparisons:.0%})\n"
          f"- {my_wins_d} my wins ({my_wins_d / comparisons:.0%})\n"
          f"\n"
          f"matlab error\n"
          f"- {pretty_float(matlab_total_error)} total error ({pretty_float(matlab_total_error / comparisons)} average)\n"
          f"- {pretty_float(matlab_total_error_s)} total s-error ({pretty_float(matlab_total_error_s / comparisons)} average)\n"
          f"- {pretty_float(matlab_total_error_d)} total d-error ({pretty_float(matlab_total_error_d / comparisons)} average)\n"
          f"\n"
          f"my error\n"
          f"- {pretty_float(my_total_error)} total error ({pretty_float(my_total_error / comparisons)} average)\n"
          f"- {pretty_float(my_total_error_s)} total s-error ({pretty_float(my_total_error_s / comparisons)} average)\n"
          f"- {pretty_float(my_total_error_d)} total d-error ({pretty_float(my_total_error_d / comparisons)} average)")


if __name__ == "__main__":
    main()


class FrenetAgent(Agent):
    def __init__(self, waypoints, **kwargs):
        super().__init__(**kwargs)

        self.waypoints = waypoints

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

        self.targets, self.spline2d = self.frenet_planner.generate_target_course([waypoint.x for waypoint in waypoints], [waypoint.y for waypoint in waypoints])

    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        pass

    def process_feedback(self, previous_state, action, state, reward):
        pass
