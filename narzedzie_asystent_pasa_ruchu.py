#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      adria
#
# Created:     17-10-2018
# Copyright:   (c) adria 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import PIL
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog



CACHE_LEFT_SLOPE = 0
CACHE_RIGHT_SLOPE = 0
CACHE_LEFT = [0, 0, 0]
CACHE_RIGHT = [0, 0, 0]

def reset_globals():

    global CACHE_LEFT_SLOPE
    global CACHE_RIGHT_SLOPE
    global CACHE_LEFT
    global CACHE_RIGHT
    CACHE_LEFT_SLOPE = 0
    CACHE_RIGHT_SLOPE = 0
    CACHE_LEFT = [0, 0, 0]
    CACHE_RIGHT = [0, 0, 0]


def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def gaussian_filter(image,kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def usun_male_komponenty(image, threshold):
    # pozbycie sie malych komponentow nie bedacych znakami
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    img_bez_komponentow = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img_bez_komponentow[output == i + 1] = 255

    return img_bez_komponentow

def canny(image,low_threshold,high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):

	mask = np.zeros_like(image)

	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def hough_lines(image):

    #hough
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 10
    max_line_gap = 5

    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def swap_canals_to_pil(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    return image

def get_vertices(image,width,height):


    if width == 960 and height == 540:
            vertices = np.array([[(width/2-width/3-100,height-50),(width/2 - width/6 +60 ,height/2+80),
            (width/2+width/6 ,height/2+80),(width/2+width/3,height-50)]],dtype=np.int32)

    elif width == 1280 and height == 720:
            vertices = np.array([[(width/2-width/3-120,height),(width/2-50 ,height/2+90),
            (width/2+100 ,height/2+90),(width/2+550,height)]],dtype=np.int32)

    elif width == 1920 and height == 1080:
            vertices = np.array([[(width/2-width/3-180,height),(width/2-100 ,height/2+90),
            (width/2+290 ,height/2+90),(width/2+850,height)]],dtype=np.int32)

    return vertices

def draw_lines(image, lines, color=[255, 0, 0], thickness=16):

    global CACHE_LEFT_SLOPE
    global CACHE_RIGHT_SLOPE
    global CACHE_LEFT
    global CACHE_RIGHT

    # DECLARE VARIABLES
    cache_weight = 0.9
    right_ys = []
    right_xs = []
    right_slopes = []

    left_ys = []
    left_xs = []
    left_slopes = []

    midpoint = image.shape[1] / 2
    bottom_of_image = image.shape[0]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # Filter lines using slope and x position
            if .35 < np.absolute(slope) <= .85:
                if slope > 0 and x1 > midpoint and x2 > midpoint:
                    right_ys.append(y1)
                    right_ys.append(y2)
                    right_xs.append(x1)
                    right_xs.append(x2)
                    right_slopes.append(slope)


                elif slope < 0 and x1 < midpoint and x2 < midpoint:
                    left_ys.append(y1)
                    left_ys.append(y2)
                    left_xs.append(x1)
                    left_xs.append(x2)
                    left_slopes.append(slope)


    if right_ys:
        right_index = right_ys.index(min(right_ys))
        right_x1 = right_xs[right_index]
        right_y1 = right_ys[right_index]
        right_slope = np.median(right_slopes)
        if CACHE_RIGHT_SLOPE != 0:
            right_slope = right_slope + (CACHE_RIGHT_SLOPE - right_slope) * cache_weight

        right_x2 = int(right_x1 + (bottom_of_image - right_y1) / right_slope)

        if CACHE_RIGHT_SLOPE != 0:
            right_x1 = int(right_x1 + (CACHE_RIGHT[0] - right_x1) * cache_weight)
            right_y1 = int(right_y1 + (CACHE_RIGHT[1] - right_y1) * cache_weight)
            right_x2 = int(right_x2 + (CACHE_RIGHT[2] - right_x2) * cache_weight)

        CACHE_RIGHT_SLOPE = right_slope
        CACHE_RIGHT = [right_x1, right_y1, right_x2]

        cv2.line(image, (right_x1, right_y1), (right_x2, bottom_of_image), color, thickness)

    # DRAW LEFT LANE LINE
    if left_ys:
        left_index = left_ys.index(min(left_ys))
        left_x1 = left_xs[left_index]
        left_y1 = left_ys[left_index]
        left_slope = np.median(left_slopes)
        if CACHE_LEFT_SLOPE != 0:
            left_slope = left_slope + (CACHE_LEFT_SLOPE - left_slope) * cache_weight

        left_x2 = int(left_x1 + (bottom_of_image - left_y1) / left_slope)

        if CACHE_LEFT_SLOPE != 0:
            left_x1 = int(left_x1 + (CACHE_LEFT[0] - left_x1) * cache_weight)
            left_y1 = int(left_y1 + (CACHE_LEFT[1] - left_y1) * cache_weight)
            left_x2 = int(left_x2 + (CACHE_LEFT[2] - left_x2) * cache_weight)

        CACHE_LEFT_SLOPE = left_slope
        CACHE_LEFT = [left_x1, left_y1, left_x2]

        cv2.line(image, (left_x1, left_y1), (left_x2, bottom_of_image), color, thickness)

def process_image(image):

    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)

    width_test = image.shape[1]
    height_test = image.shape[0]

    imshape = image.shape
    vertices = get_vertices(image,width_test,height_test)

    #zamien na skale szarosci
    gray = grayscale(image)
    gray[gray < 120] = 0
    #filt gaussa
    kernel_size = 5
    gauss = gaussian_filter(gray,kernel_size)

    #edge
    low_threshold = 25
    high_threshold = 75
    edges = canny(gauss,low_threshold,high_threshold)

    test_roi = region_of_interest(edges,vertices)

    lines = hough_lines(test_roi)

    hough_image = np.zeros((*test_roi.shape, 3), dtype=np.uint8)
    draw_lines(hough_image, lines)


    ended = cv2.addWeighted(image, 0.8, hough_image, 1, 0)

    return ended

def select_image():

    reset_globals()

    global panelA, panelB

    path = filedialog.askopenfilename()

    if len(path) > 0:

        image = cv2.imread(path)
        ended = process_image(image)

        image = cv2.resize(image, (480,270), interpolation=cv2.INTER_AREA)
        ended = cv2.resize(ended, (480,270), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        ended = Image.fromarray(ended)

        image = ImageTk.PhotoImage(image)
        ended = ImageTk.PhotoImage(ended)



        if panelA is None or panelB is None:


            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)


            panelB = Label(image=ended)
            panelB.image = ended
            panelB.pack(side="right", padx=10, pady=10)

        else:

            panelA.configure(image=image)
            panelB.configure(image=ended)
            panelA.image = image
            panelB.image = ended



root = Tk()
panelA = None
panelB = None
T = Text(root, height=1, width=30)
T.pack()
T.insert(END, "Narzędzie testowania aplikacji")
btn = Button(root, text="Wybierz obraz", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
root.mainloop()






