import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a function to evaluate a Bézier curve at parameter t
def evaluate_bezier_curve(t, control_points):
    n = len(control_points) - 1
    result = np.zeros_like(control_points[0])
    for i, point in enumerate(control_points):
        result += point * np.math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)
    return result

# Define a function to parse and evaluate the Bézier curve from a JSON string
def parse_and_evaluate(json_string):
    data = json.loads(json_string)
    bezier_curve = [eval(data['latex']) for _ in range(2)]
    return bezier_curve

# Define a function to update the plot with the current cursor position
def update_plot(frame):
    t = frame / frames_total
    cursor_position = evaluate_bezier_curve(t, bezier_curve)
    cursor.set_data(cursor_position[0], cursor_position[1])
    return cursor,

# List of JSON strings representing Bézier curves
json_strings = [
    "{'id': 'expr-4555', 'latex': '((1-t)((1-t)((1-t)862.877524+t862.081326)+t((1-t)862.081326+t861.991949))+t((1-t)((1-t)862.081326+t861.991949)+t((1-t)861.991949+t862.555680)),                (1-t)((1-t)((1-t)12.460082+t13.211212)+t((1-t)13.211212+t12.805744))+t((1-t)((1-t)13.211212+t12.805744)+t((1-t)12.805744+t11.000000)))', 'color': '#2464b4', 'secret': True}"
    
]

# Parse and evaluate the Bézier curves
bezier_curves = [parse_and_evaluate(json_string) for json_string in json_strings]

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 1000)  # Adjust xlim as needed
ax.set_ylim(0, 1000)  # Adjust ylim as needed
cursor, = ax.plot([], [], marker='o', color='red')

# Animation parameters
frames_total = 100

# Animate the cursor along each Bézier curve
anim = FuncAnimation(fig, update_plot, frames=frames_total, interval=50, blit=True)

plt.show()
