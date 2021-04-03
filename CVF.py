from matplotlib import pyplot as plt
import json
import pprint as pp
import numpy as np
import csv

# TOTAL TIME
T = 1

# LOAD DATA FROM IMAGE
with open("Kiwi/kiwi.json") as jsonFile:
    data = json.load(jsonFile)
    curves = data['curves']

# REMOVE UNWANTED DATA (MOVETO and POLYCLOSE)
for dict in curves:
    if (dict['code'] == 'M' or dict['code'] == 'Z'):
        curves.remove(dict)

# PRINTS THE DATAPOINTS
pp.pprint(curves)

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
            X = ((1 - tc)**3)*x0 + 3*((1 - tc)**2)*tc*x1 + 3*(1 - tc)*(tc**2)*x2 + (tc**3)*x
            Y = ((1 - tc)**3)*y0 + 3*((1 - tc)**2)*tc*y1 + 3*(1 - tc)*(tc**2)*y2 + (tc**3)*y
        elif curves[curveIndex]['code'] == 'L':
            X = (1 - tc)*x0 + tc*x
            Y = (1 - tc)*y0 + tc*y
        return [X, Y]
    else:
        print("Input out of domain")

# TESTING
t = 0
while t < T:
    with open("../MathWork/dataPoints.csv", 'a', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow(f(t))
    t = t + 0.001