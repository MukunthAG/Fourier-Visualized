# pyright: reportUndefinedVariable=false


from ast import increment_lineno
from os import O_PATH
from manimlib.imports import *
from matplotlib import pyplot as plt
import json
import pprint as pp
import numpy as np

# TOTAL TIME
T = 1

# LOAD FILE
JSON = "Bhavya/BhavyaFull.json"
NV = 170
SlowDown = 0.1

# LOAD DATA FROM IMAGE
with open(JSON) as jsonFile:
    data = json.load(jsonFile)
    curves = data['curves']

# REMOVE UNWANTED DATA (MOVETO and POLYCLOSE)
for dict in curves:
    if (dict['code'] == 'M' or dict['code'] == 'Z'):
        curves.remove(dict)

# PRINTS THE DATAPOINTS
# pp.pprint(curves)

# COMPLEX FUNCTION
def f(t):
    if t < T and t >= 0:
        curveIndex = "String"
        intervals = np.linspace(0, T, len(curves) + 1)
        for count, time in enumerate(intervals):
            if time > t:
                curveIndex = count - 1
                break
            elif time == t:
                curveIndex = count
                break
        t0 = intervals[curveIndex]
        t1 = intervals[curveIndex + 1]
        tc = (t - t0)/(t1 - t0)
        x0 = curves[curveIndex]['x0']
        y0 = curves[curveIndex]['y0']
        x = curves[curveIndex]['x']
        y = curves[curveIndex]['y']
        if curves[curveIndex]['code'] == 'C':
            x1 = curves[curveIndex]['x1']
            y1 = curves[curveIndex]['y1']
            x2 = curves[curveIndex]['x2']
            y2 = curves[curveIndex]['y2']
            Xout = ((1 - tc)**3)*x0 + 3*((1 - tc)**2)*tc*x1 + 3*(1 - tc)*(tc**2)*x2 + (tc**3)*x
            Yout = ((1 - tc)**3)*y0 + 3*((1 - tc)**2)*tc*y1 + 3*(1 - tc)*(tc**2)*y2 + (tc**3)*y
        elif curves[curveIndex]['code'] == 'L' or curves[curveIndex]['code'] == 'V' or curves[curveIndex]['code'] == 'H':
            Xout = (1 - tc)*x0 + tc*x
            Yout = (1 - tc)*y0 + tc*y
        else:
            print("ERRORRRRRRRRRRRRRRRRRRRRRRRRR!!!!!")
            print(curves[curveIndex]['code'])
        return [Xout, Yout] 
    else:
        print("Input out of domain")

def scaled_points(scaleX, del_t):
    t = 0
    points = []
    for t in np.arange(0, 1, del_t):
        points.append(f(t))

    x = [i[0] for i in points]
    y = [j[1] for j in points]
    slope = max(y)/max(x)
    sX = [scaleX*(i/max(x)) for i in x]
    sY = [slope*scaleX*(j/max(y)) for j in y]
    X = [i - (max(sX)/2) for i in sX]
    Y = [-1*(j - (max(sY)/2)) for j in sY]  # FLIP IS TURNED ON!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 
    return list(map(list,zip(X, Y)))

def get_freqs(no_of_vectors=99):
    n = no_of_vectors
    all_freqs = list(range(n // 2, -n // 2, -1))
    all_freqs.sort(key=abs)
    return all_freqs

def compute_integral(del_t=0.0001):
    path = scaled_points(5, del_t)
    complex_path = [p[0] + 1j*p[1] for p in path]
    intervals = np.arange(0, 1, del_t)
    freqs = get_freqs(no_of_vectors=NV)
    cn = []
    for nu in freqs:
        integral = np.array(
            [
                np.exp(-2*PI*nu*t*1j)*ft
                for t, ft in zip(intervals, complex_path)
            ]
        ).sum()*del_t
        cn.append(integral)
    return cn

# pp.pprint(len(compute_integral(del_t=0.001)))

class OpeningScene(Scene):
    def construct(self):
        title = TextMobject("Fourier Visualized")
        title.move_to(0.5*UP)
        title.scale(2)
        title.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE) 
        self.play(
            Write(title)
        )
        self.wait(2.5)
        self.play(
            FadeOut(title),
            run_time = 1.5
        )
        self.wait()

cn = compute_integral()

class FourierSketcher(Scene):
    CONFIG = {
        "no_of_vectors": 5,
        "phase": [np.log(c).imag for c in cn],
        "frequency": get_freqs(no_of_vectors=NV),
        "length": [np.exp(np.log(c).real) for c in cn]
    }
    def setup(self):
        self.clock = ValueTracker(0)
        self.clock.add_updater(
            lambda m, dt: m.increment_value(SlowDown*dt)
        )
        self.add(self.clock)
    
    def get_timer(self):
        return self.clock.get_value()

    def create_vector_group(self):
        vector_group = VGroup()
        freq = self.frequency
        phase = self.phase
        len = self.length
        last_vector = None
        for nu, phi, mag in zip(freq, phase, len):
            vector = Vector(direction=RIGHT, color=YELLOW)
            vector.nu = nu
            vector.phi = phi
            vector.rotate(phi, about_point=ORIGIN)
            vector.set_length(mag)
            if last_vector:
                vector.start_from = last_vector.get_end
            else:
                vector.start_from = VectorizedPoint(ORIGIN).get_location
            vector.add_updater(self.vector_rotater)
            vector_group.add(vector)
            last_vector = vector
        return vector_group
    
    def vector_rotater(self, vector, dt):
        t = self.get_timer()
        vector.set_angle((vector.phi) + 2*PI*(vector.nu)*t)
        vector.shift(vector.start_from() - vector.get_start())

    def sum_of_vectors(self, t):
        return reduce(
            op.add,
            [
            complex_to_R3(
                mag*np.exp((phi + 2*PI*nu*t)*1j) 
            )
            for mag, phi, nu in zip(self.length, self.phase, self.frequency)
            ]
        )
    
    def get_vector_sum_path(self, color=BLUE):
        path = ParametricFunction(
            self.sum_of_vectors,
            t_min=0,
            t_max=1,
            color=color,
            step_size=0.001,
        )
        return path

    def get_drawn_path_alpha(self):
        return self.get_timer()

    def get_drawn_path(self, stroke_width=None, **kwargs):
        if stroke_width is None:
            stroke_width = 2
        path = self.get_vector_sum_path(**kwargs)
        broken_path = CurvesAsSubmobjects(path)
        broken_path.curr_time = 0

        def update_path(path, dt):
            alpha = self.get_drawn_path_alpha()
            n_curves = len(path)
            for a, sp in zip(np.linspace(0, 1, n_curves), path):
                b = alpha - a
                if b < 0:
                    width = 0
                else:
                    width = stroke_width * (1 - (b % 1))
                sp.set_stroke(width=width)
            path.curr_time += dt
            return path

        broken_path.set_color(BLUE)
        broken_path.add_updater(update_path)
        return broken_path

    def construct(self):
        vectors = self.create_vector_group()
        drawn_path = self.get_drawn_path()
        self.add(vectors)
        self.add(drawn_path)
        self.wait(60)
