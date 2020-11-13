import math

from library.actors import DynamicActorConstants

M2PX = 16  # pixels per metre

car_constants = DynamicActorConstants(
    length=M2PX * 4.5,  # [4.5 m]
    width=M2PX * 1.75,  # [1.75 m]
    wheelbase=M2PX * 3,  # [3 m]
    track=M2PX * 1.75,  # [1.75 m]
    min_velocity=0,  # [0 m/s]
    max_velocity=M2PX * 9,  # [0 m/s]
    min_throttle=M2PX * -9,  # [-9 m/ss]
    max_throttle=M2PX * 9,  # [9 m/ss]
    min_steering_angle=-(math.pi * 0.2),  # [35 degrees right]
    max_steering_angle=math.pi * 0.2  # [35 degrees left]
)

pedestrian_constants = DynamicActorConstants(
    length=M2PX * 0.65625,
    width=M2PX * 0.875,
    wheelbase=M2PX * 0.328125,
    track=M2PX * 0.875,
    min_velocity=0,
    max_velocity=M2PX * 1.4,
    min_throttle=-(M2PX * 1.4),
    max_throttle=M2PX * 1.4,
    min_steering_angle=-(math.pi * 0.5),
    max_steering_angle=(math.pi * 0.5)
)

bus_constants = DynamicActorConstants(
    length=M2PX * 12,
    width=M2PX * 2.55,
    wheelbase=M2PX * 16.875,
    track=M2PX * 2.55,
    min_velocity=0,
    max_velocity=M2PX * 6.75,
    min_throttle=-(M2PX * 6.75),
    max_throttle=M2PX * 6.75,
    min_steering_angle=-(math.pi * 0.16),
    max_steering_angle=(math.pi * 0.16)
)

bicycle_constants = DynamicActorConstants(
    length=M2PX * 2.25,
    width=M2PX * 0.875,
    wheelbase=M2PX * 2.025,
    track=M2PX * 0.875,
    min_velocity=0,
    max_velocity=M2PX * 4.5,
    min_throttle=-(M2PX * 4.5),
    max_throttle=M2PX * 4.5,
    min_steering_angle=-(math.pi * 0.5),
    max_steering_angle=(math.pi * 0.5)
)
