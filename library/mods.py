import math

from gym.envs.classic_control.rendering import LineStyle, glEnable, glLineStipple, GL_LINE_STIPPLE, FilledPolygon, PolyLine


class FactoredLineStyle(LineStyle):
    def __init__(self, style, factor):
        super().__init__(style)
        self.factor = factor

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(self.factor, self.style)


def make_circle(x, y, radius, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((x + math.cos(ang)*radius, y + math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
