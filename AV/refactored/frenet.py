"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""
import bisect
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from matplotlib.ticker import MaxNLocator

from AV.refactored.cavgym import DynamicBodyState, DynamicBody, Point, DynamicBodyConstants

SIM_LOOP = 500

# Parameter
MAX_VELOCITY = 50.0 / 3.6  # maximum velocity [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]: (2 m/s) / (1 s) = 2 m/ss
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_VELOCITY = 30.0 / 3.6  # target velocity [m/s]
D_T_S = 5.0 / 3.6  # target velocity sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target velocity
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class Spline:
    def __init__(self, points):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.points = points

        self.nx = len(self.points)  # dimension of x
        h = np.diff([point.x for point in self.points])

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.points[i + 1].y - self.points[i].y) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None
        """
        if t < self.points[0].x:
            return None
        elif t > self.points[-1].x:
            return None

        i = self.__search_index(t)
        dx = t - self.points[i].x
        result = self.points[i].y + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """
        if t < self.points[0].x:
            return None
        elif t > self.points[-1].x:
            return None

        i = self.__search_index(t)
        dx = t - self.points[i].x
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """
        if t < self.points[0].x:
            return None
        elif t > self.points[-1].x:
            return None

        i = self.__search_index(t)
        dx = t - self.points[i].x
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect([point.x for point in self.points], x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.points[i + 2].y - self.points[i + 1].y) / h[i + 1] - 3.0 * (self.points[i + 1].y - self.points[i].y) / h[i]
        return B


class Spline2D:
    def __init__(self, points):
        self.s = self.__calc_s(points)
        self.sx = Spline([Point(x=s, y=point.x) for s, point in zip(self.s, points)])
        self.sy = Spline([Point(x=s, y=point.y) for s, point in zip(self.s, points)])

    def __calc_s(self, points):
        dx = np.diff([point.x for point in points])
        dy = np.diff([point.y for point in points])
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        return Point(
            x=self.sx.calc(s),
            y=self.sy.calc(s)
        )

    def calc_curvature(self, s):
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
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

        A = np.array([[time ** 3, time ** 4, time ** 5], [3 * time ** 2, 4 * time ** 3, 5 * time ** 4], [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2, vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
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

        A = np.array([[3 * time ** 2, 4 * time ** 3], [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
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
        self.t = []  # time: self.t[i] = i * DT where i >= 0 is integer
        self.d = []  # lateral position
        self.d_d = []  # lateral velocity
        self.d_dd = []  # lateral acceleration
        self.d_ddd = []
        self.s = []  # longitudinal position
        self.s_d = []  # velocity
        self.s_dd = []  # acceleration
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0  # cost

        self.points = []
        self.yaw = []
        self.ds = []
        self.c = []  # curvature


def calc_frenet_paths(state):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            frenet_path = FrenetPath()

            lat_qp = QuinticPolynomial(state.lateral_position, state.lateral_velocity, state.lateral_acceleration, di, 0.0, 0.0, Ti)

            frenet_path.t = [t for t in np.arange(0.0, Ti, DT)]
            frenet_path.d = [lat_qp.calc_point(t) for t in frenet_path.t]
            frenet_path.d_d = [lat_qp.calc_first_derivative(t) for t in frenet_path.t]
            frenet_path.d_dd = [lat_qp.calc_second_derivative(t) for t in frenet_path.t]
            frenet_path.d_ddd = [lat_qp.calc_third_derivative(t) for t in frenet_path.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_VELOCITY - D_T_S * N_S_SAMPLE, TARGET_VELOCITY + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(frenet_path)
                lon_qp = QuarticPolynomial(state.longitudinal_position, state.velocity, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in frenet_path.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in frenet_path.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in frenet_path.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in frenet_path.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target velocity
                ds = (TARGET_VELOCITY - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(frenet_paths, spline2d):
    for frenet_path in frenet_paths:

        # calc global positions
        for si, di in zip(frenet_path.s, frenet_path.d):
            i_point = spline2d.calc_position(si)
            if i_point.x is None:
                break
            i_yaw = spline2d.calc_yaw(si)
            fx = i_point.x + di * math.cos(i_yaw + math.pi / 2.0)
            fy = i_point.y + di * math.sin(i_yaw + math.pi / 2.0)
            frenet_path.points.append(Point(x=fx, y=fy))

        # calc yaw and ds
        for point, next_point in zip(frenet_path.points, frenet_path.points[1:]):
            dx = next_point.x - point.x
            dy = next_point.y - point.y
            frenet_path.yaw.append(math.atan2(dy, dx))
            frenet_path.ds.append(math.hypot(dx, dy))

        frenet_path.yaw.append(frenet_path.yaw[-1])
        frenet_path.ds.append(frenet_path.ds[-1])

        # calc curvature
        for yaw, ds, next_yaw in zip(frenet_path.yaw, frenet_path.ds, frenet_path.yaw[1:]):
            frenet_path.c.append((next_yaw - yaw) / ds)

    return frenet_paths


def check_collision(frenet_path, obstacles):
    for obstacle in obstacles:
        collision = any([(path_point.x - obstacle.x) ** 2 + (path_point.y - obstacle.y) ** 2 <= ROBOT_RADIUS ** 2 for path_point in frenet_path.points])
        if collision:
            return False
    return True


def check_paths(frenet_paths, obstacles):
    ok_frenet_paths = []
    for frenet_path in frenet_paths:
        if any([v > MAX_VELOCITY for v in frenet_path.s_d]):  # Max velocity check
            continue
        elif any([abs(a) > MAX_ACCEL for a in frenet_path.s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in frenet_path.c]):  # Max curvature check
            continue
        elif not check_collision(frenet_path, obstacles):
            continue

        ok_frenet_paths.append(frenet_path)

    return ok_frenet_paths


def frenet_optimal_planning(spline2d, state, obstacles):
    frenet_paths = calc_frenet_paths(state)
    frenet_paths = calc_global_paths(frenet_paths, spline2d)
    frenet_paths = check_paths(frenet_paths, obstacles)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for frenet_path in frenet_paths:
        if min_cost >= frenet_path.cf:
            min_cost = frenet_path.cf
            best_path = frenet_path

    return best_path


def generate_target_course(points):
    spline2d = Spline2D(points)  # what is spline2d.s?

    targets = [
        Target(
            point=spline2d.calc_position(spline2d_point),
            yaw=spline2d.calc_yaw(spline2d_point),
            curvature=spline2d.calc_curvature(spline2d_point)
        ) for spline2d_point in np.arange(0, spline2d.s[-1], 0.1)  # float equivalent of range(0, spline2d.s[-1], 0.1)
    ]

    return targets, spline2d


@dataclasses.dataclass(frozen=True)
class State:
    velocity: float  # velocity [m/s]
    lateral_position: float  # lateral position [m]
    lateral_velocity: float  # lateral velocity [m/s]
    lateral_acceleration: float  # lateral acceleration [m/s]
    longitudinal_position: float  # longitudinal position


@dataclasses.dataclass(frozen=True)
class Target:
    point: Point
    yaw: float
    curvature: float


def make_point(spline2d, state):
    point = spline2d.calc_position(state.longitudinal_position)
    # point = Point(x=spline2d.sx.calc(state.longitudinal_position), y=spline2d.sy.calc(state.longitudinal_position))
    # point.x = spline2d.sx.points[i].y +
    #     spline2d.sx.b[i] * (state.longitudinal_position - spline2d.sx.points[i].x) +
    #     spline2d.sx.c[i] * (state.longitudinal_position - spline2d.sx.points[i].x) ** 2.0 +
    #     spline2d.sx.d[i] * (state.longitudinal_position - spline2d.sx.points[i].x) ** 3.0
    assert point.x is not None
    yaw = spline2d.calc_yaw(state.longitudinal_position)
    # yaw = math.atan2(spline2d.sy.calcd(state.longitudinal_position), spline2d.sx.calcd(state.longitudinal_position))
    fx = point.x + state.lateral_position * math.cos(yaw + math.pi / 2.0)
    fy = point.y + state.lateral_position * math.sin(yaw + math.pi / 2.0)
    return Point(x=fx, y=fy)


def make_state(body_state):
    return State(
        velocity=body_state.velocity,
        lateral_position=2.0,  # FrenetPath.d[0] == lat_qp.calc_point(0.0) == lat_qp.a0 + 0.0 + 0.0 + 0.0 + 0.0 == lat_qp.a0 == xs == state.lateral_position
        lateral_velocity=0.0,  # FrenetPath.d_d[0] == lat_qp.calc_first_derivative(0.0) == self.a1 + 0.0 + 0.0 + 0.0 + 0.0 == lat_qp.a1 == vxs == state.lateral_velocity
        lateral_acceleration=0.0,  # FrenetPath.d_dd[0] == lat_qp.calc_second_derivative(0.0) == 2 * lat_qp.a2 + 0.0 + 0.0 + 0.0 == 2 * lat_qp.a2 == axs == state.lateral_acceleration
        longitudinal_position=0.0  # FrenetPath.s[0] == lon_qp.calc_point(0.0) == lon_qp.a0 + 0.0 + 0.0 + 0.0 + 0.0 == lon_qp.a0 == xs == state.longitudinal_position
    )


def test(path, body_state):
    print(path.points[0], body_state.position)
    print(path.s_d[0], body_state.velocity)
    print(path.yaw[0], body_state.orientation)


def main():
    print(__file__ + " start!!")

    # way points
    target_waypoints = [
        Point(x=0.0, y=0.0),
        Point(x=10.0, y=-6.0),
        Point(x=20.5, y=5.0),
        Point(x=35.0, y=6.5),
        Point(x=70.5, y=0.0)
    ]
    # obstacle lists
    obstacles = [
        Point(x=20.0, y=10.0),
        Point(x=30.0, y=6.0),
        Point(x=30.0, y=8.0),
        Point(x=35.0, y=8.0),
        Point(x=50.0, y=3.0)
    ]

    targets, target_spline2d = generate_target_course(target_waypoints)

    # initial state
    state = State(
        velocity=10.0 / 3.6,
        lateral_position=2.0,
        lateral_velocity=0.0,
        lateral_acceleration=0.0,
        longitudinal_position=0.0
    )

    area = 20.0  # animation area length [m]

    # body_state = DynamicBodyState(
    #     position=Point(x=1.3527664541170896, y=1.4730997660089002),
    #     velocity=10.0 / 3.6,
    #     orientation=-0.7444386098364741
    # )
    body_state = DynamicBodyState(
        position=Point(x=0.0, y=0.0),
        velocity=10.0 / 3.6,
        orientation=0.0
    )

    body_constants = DynamicBodyConstants(
        length=4.5,
        width=1.75,
        wheelbase=3,
        min_velocity=0,
        max_velocity=9,
        min_throttle=-9,
        max_throttle=9,
        min_steering_angle=-(math.pi * 0.2),
        max_steering_angle=math.pi * 0.2
    )

    body = DynamicBody(init_state=body_state, constants=body_constants)

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(target_spline2d, state, obstacles)
        # path = frenet_optimal_planning(target_spline2d, make_state(body.state), obstacles)

        body.state = DynamicBodyState(
            position=path.points[1],
            velocity=path.s_d[1],
            orientation=path.yaw[1]
        )

        # test(path, body_state)
        #
        # body_successor_state = DynamicBodyState(
        #     position=path.points[1],
        #     velocity=path.s_d[1],
        #     orientation=path.yaw[1]
        # )
        #
        # action = body.action_translation(body_successor_state, DT)
        # throttle, steering_angle = action
        # print(f"action=({throttle}, {math.degrees(steering_angle)})")
        # body.step(action, DT)

        state = State(
            velocity=path.s_d[1],
            lateral_position=path.d[1],
            lateral_velocity=path.d_d[1],
            lateral_acceleration=path.d_dd[1],
            longitudinal_position=path.s[1]
        )

        if np.hypot(path.points[1].x - targets[-1].point.x, path.points[1].y - targets[-1].point.y) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.gca().set_aspect('equal', adjustable='box')
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot([target.point.x for target in targets], [target.point.y for target in targets])  # target path (blue line)
            plt.plot([obstacle.x for obstacle in obstacles], [obstacle.y for obstacle in obstacles], "xk")  # obstacles (grey crosses)
            plt.plot([point.x for point in path.points[1:]], [point.y for point in path.points[1:]], "-or")  # current path (red dotted line)
            plt.plot(path.points[1].x, path.points[1].y, "vc")  # current position (cyan triangle)

            bb = body.bounding_box()
            bbx, bby = zip(*[bb.rear_left, bb.front_left, bb.front_right, bb.rear_right, bb.rear_left])
            plt.plot(bbx, bby)

            plt.xlim(path.points[1].x - area, path.points[1].x + area)
            plt.ylim(path.points[1].y - area, path.points[1].y + area)
            plt.title("v[km/h]:" + str(state.velocity * 3.6)[0:4])  # convert m/s to km/h
            plt.grid(True)
            plt.pause(0.0001)

            # plt.xticks(range(-20, 21))
            # plt.yticks(range(-20, 21))
            #
            # plt.pause(100)

            # plt.savefig(f"fig-{i:02}.pdf")

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
