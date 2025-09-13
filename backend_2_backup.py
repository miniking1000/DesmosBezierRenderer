import json
from collections import defaultdict

from flask import Flask
from flask_cors import CORS
from flask import request
from PyDesmos import Graph
import PyDesmos

from PIL import Image
import numpy as np
import potrace
import cv2

import multiprocessing
from time import time
import os
import sys
import getopt
import traceback

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import re

app = Flask(__name__)
CORS(app)


DYNAMIC_BLOCK = True # Automatically find the right block size
BLOCK_SIZE = 25 # Number of frames per block (ignored if DYNAMIC_BLOCK is true)
MAX_EXPR_PER_BLOCK = 7500 # Maximum lines per block, doesn't affect lines per frame (ignored if DYNAMIC_BLOCK is false)

FRAME_DIR = 'frames' # The folder where the frames are stored relative to this file
FILE_EXT = 'png' # Extension for frame files
COLOUR = '#2464b4' # Hex value of colour for graph output	

BILATERAL_FILTER = False # Reduce number of lines with bilateral filter
DOWNLOAD_IMAGES = False # Download each rendered frame automatically (works best in firefox)
USE_L2_GRADIENT = False # Creates less edges but is still accurate (leads to faster renders)
SHOW_GRID = True # Show the grid in the background while rendering
NEED_TO_CLOUD = False
MAX_LINES = 10000

port=5001
frame = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock = False)
width = multiprocessing.Value('i', 0, lock = False)
frame_latex = 0


def split_dicts_by_latex_old(data_list):
    # Create a defaultdict to hold lists of dicts based on latex expression parts
    latex_segments = defaultdict(list)

    # Iterate through each dictionary in the input list
    # data_list =
    for entry in data_list:
        # Split the latex string by ',' and use it as a key in latex_segments
        latex_parts = entry['latex'].split(',')

        # For each part, append the original dict to the corresponding list
        for part in latex_parts:
            latex_segments[part.strip()].append(entry)

    # Create the final segmented list of dictionaries
    segmented_dicts = list(latex_segments.values())

    return segmented_dicts


def split_dicts_by_latex(data_list, max_size):
    # Create a dictionary to hold segments based on latex expressions
    result = []
    new_list = []
    for i in range(len(data_list)):
        while len(new_list) <= max_size:
            new_list.append(data_list[i])

        result.append(new_list)
        new_list = []
    return result
    segments = []
    current_segment = []

    for item in data_list:
        current_segment.append(item)

        # Check the length of current segment
        if len(segments) >= max_size:
            # Append only if the segment has more than max_size dictionaries
            segments.append(current_segment)
            current_segment = []  # Reset for the next segment

    # Check if there's any remaining items in the last segment
    if current_segment:
        segments.append(current_segment)

    # Filter segments to keep only those that are larger than max_size or the last segment
    final_segments = []
    for i, segment in enumerate(segments):
        if len(segment) > max_size or (i == len(segments) - 1):
            final_segments.append(segment)

    return final_segments


def render_large_frame(expressions, frame_number):
    """
    This function will render frames with too many expressions separately and save them.
    """
    fig, ax = plt.subplots()

    # Render each expression
    for expr in expressions:
        # Extract the coordinates and color (you can use matplotlib to draw the expressions accordingly)
        latex_expr = expr['latex']['latex']
        color = expr['color']

        ax.plot([0, 1], [2, 2], color=color)  # Replace with actual expression rendering logic

    # Save the frame to a file (as PNG, similar to Desmos)
    filename = f'large_frame_{frame_number}.png'
    plt.savefig(filename)
    plt.close(fig)
    print(f'Saved large frame {frame_number} to {filename}')


def get_expressions_new(frame):
    exprid = 0
    exprs = []
    # Retrieve expressions for the given frame
    frame_exprs = frame_latex[frame]

    if len(frame_exprs) > MAX_LINES:
        # If frame has too many expressions, render it separately
        print(f'Frame {frame} has too many expressions: {len(frame_exprs)}. Rendering separately.')
        render_large_frame([{'id': f'expr-{exprid}', 'latex': expr, 'color': COLOUR, 'secret': True} for exprid, expr in enumerate(frame_exprs)], frame)
        return []  # Return empty, as it's handled separately
    else:
        # For frames within the limit, return the expressions as normal
        for expr in frame_exprs:
            exprid += 1
            exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})

    return exprs


def help():
    print('backend.py -f <source> -e <extension> -c <colour> -b -d -l -g --static --block=<block size> --maxpblock=<max expressions per block>\n')
    print('\t-h\tGet help\n')
    print('-Render options\n')
    print('\t-f <source>\tThe directory from which the frames are stored (e.g. frames)')
    print('\t-e <extension>\tThe extension of the frame files (e.g. png)')
    print('\t-c <colour>\tThe colour of the lines to be drawn (e.g. #2464b4)')
    print('\t-b\t\tReduce number of lines with bilateral filter for simpler renders')
    print('\t-d\t\tDownload rendered frames automatically (only available if rendering quick.html)')
    print('\t-l\t\tReduce number of lines with L2 gradient for quicker renders')
    print('\t-g\t\tHide the grid in the background of the graph (only available if rendering quick.html)\n')
    print('-Optimisational options\n')
    print('\t--static\t\t\t\t\tUse a static number of expressions per request block')
    print('\t--block=<block size>\t\t\t\tThe number of frames per block in dynamic blocks')
    print('\t--maxpblock=<maximum expressions per block>\tThe maximum number of expressions per block in static blocks')
    print('\t--lines=<maximum lines to actually render>')


def get_contours(filename, nudge = .33):
    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if BILATERAL_FILTER:
        median = max(10, min(245, np.median(gray)))
        lower = int(max(0, (1 - nudge) * median))
        upper = int(min(255, (1 + nudge) * median))
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        edged = cv2.Canny(filtered, lower, upper, L2gradient = USE_L2_GRADIENT)
    else:
        edged = cv2.Canny(gray, 30, 200)

    with frame.get_lock():
        frame.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])
    print('\r--> Frame %d/%d' % (frame.value, len(os.listdir(FRAME_DIR))), end='')

    return edged[::-1]


def get_trace(data):
    for i in range(len(data)):
        data[i][data[i] > 1] = 1
    bmp = potrace.Bitmap(data)
    path = bmp.trace(2, potrace.TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path


def get_latex(filename):
    latex = []
    path = get_trace(get_contours(filename))

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start
            if segment.is_corner:
                x1, y1 = segment.c
                x2, y2 = segment.end_point
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = segment.c1
                x2, y2 = segment.c2
                x3, y3 = segment.end_point
                latex.append('((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),\
                (1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))' % \
                (x0, x1, x1, x2, x1, x2, x2, x3, y0, y1, y1, y2, y1, y2, y2, y3))
            start = segment.end_point
    return latex


def get_expressions(frame):
    exprid = 0
    exprs = []
    for expr in get_latex(FRAME_DIR + '/frame%d.%s' % (frame+1, FILE_EXT)):
        exprid += 1
        exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})
    return exprs


@app.route('/')
def index():
    frame = int(request.args.get('frame'))
    if frame >= len(frame_latex):
        return {'result': None}

    block = []
    if not DYNAMIC_BLOCK:
        number_of_frames = min(frame + BLOCK_SIZE, len(os.listdir(FRAME_DIR))) - frame
        for i in range(frame, frame + number_of_frames):
            block.append(frame_latex[i])
    else:
        number_of_frames = 0
        total = 0
        i = frame
        while total < MAX_EXPR_PER_BLOCK:
            if i >= len(frame_latex):
                break
            number_of_frames += 1
            total += len(frame_latex[i])
            block.append(frame_latex[i])
            i += 1

    return json.dumps({'result': block, 'number_of_frames': number_of_frames}) # Number_of_frames is the number of newly loaded frames, not the total frames


@app.route('/init')
def init():
    return json.dumps({'height': height.value, 'width': width.value, 'total_frames': len(frame_latex), 'download_images': DOWNLOAD_IMAGES, 'show_grid': SHOW_GRID})


@app.route('/timeing')
def timeing():
    timee = request.args.get('timee')
    with open("timing.txt", "a+") as f:
        f.write("y=" + timee.replace("?"," ").replace("lines","x") + "\n")
    return json.dumps({"True": True})


if __name__ == '__main__':
    estimated_time = 0.0
    total_lines = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:e:c:bdlgr", ['static', 'block=', 'maxpblock=', 'port='])

    except getopt.GetoptError:
        print('Error: Invalid argument(s)\n')
        help()
        sys.exit(2)

    try:
        for opt, arg in opts:
            if opt == '-h':
                help()
                sys.exit()
            elif opt == '-f':
                FRAME_DIR = arg
            elif opt == '-e':
                FILE_EXT = arg
            elif opt == '-c':
                COLOUR = arg
            elif opt == '-b':
                BILATERAL_FILTER = True
            elif opt == '-d':
                DOWNLOAD_IMAGES = True
            elif opt == '-l':
                USE_L2_GRADIENT = True
            elif opt == '-r':
                NEED_TO_CLOUD = True
            elif opt == '-g':
                SHOW_GRID = False
            elif opt == '--static':
                DYNAMIC_BLOCK = False
            elif opt == '--block':
                BLOCK_SIZE = int(arg)
            elif opt == '--lines':
                MAX_LINES = int(arg)
            elif opt == '--port':
                port = int(arg)
            elif opt == '--maxpblock':
                MAX_EXPR_PER_BLOCK = int(arg)
        frame_latex = range(len(os.listdir(FRAME_DIR)))
        #frame_latex = range(10000)

    except TypeError:
        print('Error: Invalid argument(s)\n')
        help()
        sys.exit(2)

    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        print('Desmos Bezier Renderer')
        print('Junferno 2021')
        print('https://github.com/kevinjycui/DesmosBezierRenderer')

        print('-----------------------------')

        print('Processing %d frames... Please wait for processing to finish before running on frontend\n' % len(os.listdir(FRAME_DIR)))

        start = time()

        try:
            frame_latex = pool.map(get_expressions, frame_latex)

        except cv2.error as e:
            print('[ERROR] Unable to process one or more files. Remember image files should be named <DIRECTORY>/frame<INDEX>.<EXTENSION> where INDEX represents the frame number starting from 1 and DIRECTORY and EXTENSION are defined by command line arguments (e.g. frames/frame1.png). Please check if:\n\tThe files exist\n\tThe files are all valid image files\n\tThe name of the files given is correct as per command line arguments\n\tThe program has the necessary permissions to read the file.\n\nUse backend.py -h for further documentation\n')            

            print('-----------------------------')

            print('Full error traceback:\n')
            traceback.print_exc()
            sys.exit(2)

        numbers = []
        times = []
        for i in range(len(frame_latex)):
            estimated_time += 0.0000017 * (len(frame_latex[i]) ** 2)
            total_lines += len(frame_latex[i])
            numbers.append(len(frame_latex[i]))
            times.append(0.0000017 * (len(frame_latex[i]) ** 2))
        indexed_numbers = list(enumerate(numbers))
        sorted_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)
        top_15_with_indexes = sorted_numbers[:15]
        print("\n")
        new_frame_latex = []
        for index, number in top_15_with_indexes:
            print(f"dangerous! frame {index} might be undone, its lines are {number}")
            #print(frame_latex[index][0]['latex'])

            splited = [frame_latex[index][i:i + MAX_LINES] for i in range(0, len(frame_latex[index]), MAX_LINES)]

            #splited = split_dicts_by_latex(frame_latex[index], 1000)
            print(len(splited), f" <- len splited for index {index}")
            for x in splited:
                new_frame_latex.append(x)
            #with Graph("x") as G:
            #    f, x, y = G.f, G.x, G.y
            #    for i in range(1000):
            #        G.append(frame_latex[index][i]['latex'])
            #    G.new_table({x: numbers, y: times})
        print(len(new_frame_latex))
        del frame_latex
        frame_latex = new_frame_latex

        print('\r--> Processing complete in %.1f seconds' % (time() - start),
              '\n--> Estimated time to complete is %.1f seconds' % estimated_time,
              '\n--> Total lines is %s lines' % total_lines)
        if NEED_TO_CLOUD:
            with Graph("todraw") as G:
                f, x, y = G.f, G.x, G.y
                f[x] = 0.0000017*(x ** 2)
                G.new_table({x: numbers, y: times})  # where x, y = G.x, G.y, OR

        app.run(port=port)
