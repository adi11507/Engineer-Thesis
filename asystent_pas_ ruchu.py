#-------------------------------------------------------------------------------
# Name:        Asytent detekcji pasa ruchu
# Purpose:
#
# Author:      adrian
#
# Created:     17-10-2018
# Copyright:   (c) adrian 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


CACHE_LEFT_SLOPE = 0
CACHE_RIGHT_SLOPE = 0
CACHE_LEFT = [0, 0, 0]
CACHE_RIGHT = [0, 0, 0]
central_point = 0
left_distance = 0
right_distance = 0
dotted_line_right = 0
dotted_line_left = 0
solid_line_left = 0
solid_line_right = 0
frames = 0
pom1 = 0
pom2 = 0

# wyzerowanie zmiennych globalnych


def reset_globals():

    global CACHE_LEFT_SLOPE
    global CACHE_RIGHT_SLOPE
    global CACHE_LEFT
    global CACHE_RIGHT
    CACHE_LEFT_SLOPE = 0
    CACHE_RIGHT_SLOPE = 0
    CACHE_LEFT = [0, 0, 0]
    CACHE_RIGHT = [0, 0, 0]
    frames = 0


def grayscale(image):  # konwersja na skale szarosci
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_filter(image,kernel_size): # zastosowanie filtru gaussa
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image,low_threshold,high_threshold):  # wykrywanie krawedzi
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(image, vertices):   # definicja regionu zainteresowania

    # pusta maska
	mask = np.zeros_like(image)

	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

    # wypelnienie w zadanych wierzcholkach
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


def hough_lines(image):

    # znajdz proste na obrazie o zadanych parametrach
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 10
    max_line_gap = 5

    # zwrocenie do macierzy wykrytych linii
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def get_vertices(image,width,height): # wierzcholki ze wzgledu na rozdzielczosc

    if width == 960 and height == 540:
            vertices = np.array([[(width/2-width/3-50,height-50),(width/2 - width/6 +60 ,height/2+100),
            (width/2+width/6 ,height/2+100),(width/2+width/3,height-50)]],dtype=np.int32)

    elif width == 1280 and height == 720:
            vertices = np.array([[(width/2-width/3-120,height),(width/2-50 ,height/2+90),
            (width/2+100 ,height/2+90),(width/2+550,height)]],dtype=np.int32)

    elif width == 1920 and height == 1080:
            vertices = np.array([[(width/2-width/3-180,height),(width/2-100 ,height/2+110),
            (width/2+150 ,height/2+110),(width/2+750,height)]],dtype=np.int32)

    return vertices


def draw_lines(image, lines, color=[255, 0, 0], thickness=16):

    global CACHE_LEFT_SLOPE,CACHE_RIGHT_SLOPE,CACHE_LEFT,CACHE_RIGHT
    global right_distance,left_distance
    global central_point
    global dotted_line_left, dotted_line_right
    global solid_line_left, solid_line_right
    global frames
    global pom1, pom2
    pixel_in_cm = 0.0345

    # definiowanie potrzebnych wartosci
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
            # oblicz nachylenie dla kazdej znalezionej linii
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # Filter lines using slope and x position
            if .35 < np.absolute(slope) <= .85:
                # nachylenie wieksze od 0 i po prawej stronie od punktu
                # srodkowego = linia prawa
                if slope > 0 and x1 > midpoint and x2 > midpoint:
                    right_ys.append(y1)
                    right_ys.append(y2)
                    right_xs.append(x1)
                    right_xs.append(x2)
                    right_slopes.append(slope)

                # nachylenie mniejsze od 0 i po lewej stronie od punktu
                # srodkowego = linia lewa
                elif slope < 0 and x1 < midpoint and x2 < midpoint:
                    left_ys.append(y1)
                    left_ys.append(y2)
                    left_xs.append(x1)
                    left_xs.append(x2)
                    left_slopes.append(slope)

    # narzedzie do wykrycia linii ciaglej badz przerywanej
    # ze wzgledu na wielkosc dystansu pomiedzy znalezionymi liniami
    # oraz ustawienie zmiennych pomocniczyc do wyswietlenia informacji
    # na 1 gdy zachodzi dany warunek

    if right_ys and (frames % 50) == 0:
        if abs(right_ys[1]-right_ys[0]) > 80 and abs(right_ys[1]-right_ys[0]) < 120:
            pom1 = 1

        elif abs(right_ys[1]-right_ys[0]) > 0 and abs(right_ys[1]-right_ys[0]) <15 and (frames % 50) == 0:
            pom1 = 2

    if left_ys:
        if abs(left_ys[1] - left_ys[0]) > 80 and abs(left_ys[1] - left_ys[0]) < 120 and (frames % 50) == 0:
            pom2 = 1

        elif abs(left_ys[1] - left_ys[0]) > 0 and abs(left_ys[1] - left_ys[0])< 15 and (frames % 50) == 0:
            pom2 = 2


# narysuj linie prawa
    if right_ys:
        # znajdz indeks minimalnego punktu prostej prawej
        right_index = right_ys.index(min(right_ys))
        # znajdz punkty odnosnie tego indeksu
        right_x1 = right_xs[right_index]
        right_y1 = right_ys[right_index]
        # oblicz mediana od wszystkich nachylen
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

        # obliczenie dystansu od prawej linii
        right_distance_1 = right_x1 - midpoint
        right_distance_2 = right_x2 - midpoint
        right_distance = np.mean(right_distance_1 + right_distance_2)
        right_distance = (right_distance * pixel_in_cm) / 10
        # jesli mniejszy niz 1 metr strzalka wyrownaj do lewej
        if right_distance < 1:
            cv2.arrowedLine(image,(right_x1 + 200, right_y1 +50) , (right_x1 +120, right_y1 +50), (255,255,255), 7)
            cv2.putText(image, "Wyrownaj do lewej!", (right_x1+100, right_y1+20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        cv2.line(image, (right_x1, right_y1), (right_x2, bottom_of_image), color, thickness)


    # narysuj linie lewa
    if left_ys:
        # znajdz indeks minimalnej linii lewej
        left_index = left_ys.index(min(left_ys))
        # zapisz punkty prostej o tym indeksie
        left_x1 = left_xs[left_index]
        left_y1 = left_ys[left_index]
        # oblicz mediane nachylen
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

        # oblicz dystans do lewej linii
        left_distance_1 = midpoint - left_x1
        left_distance_2 = midpoint - left_x2
        left_distance = np.mean(left_distance_1+left_distance_2)
        left_distance = (left_distance * pixel_in_cm) / 10

        # jesli mniejszy niz 1 metr wyrownaj do prawej
        if left_distance < 1:
            cv2.arrowedLine(image,(left_x1 - 200, left_y1+50) , (left_x1 -120, left_y1 +50), (255,255,255), 7)
            cv2.putText(image, "Wyrownaj do prawej!", (left_x1 -100, left_y1+20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # rysuj linie na obrazie wyjsciowym
        cv2.line(image, (left_x1, left_y1), (left_x2, bottom_of_image), color, thickness)



def process_image(image):

    global central_point
    global right_distance,left_distance
    global dotted_line_left,dotted_line_right
    global solid_line_left, solid_line_right

    # zmien rozdzielczosc, przykladowo na takie parametry ekranu
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)

    width_test = image.shape[1]
    height_test = image.shape[0]

    imshape = image.shape
    # nadaj wierzcholki
    vertices = get_vertices(image,width_test,height_test)

    #zamien na skale szarosci
    gray = grayscale(image)
    gray[gray < 120] = 0
    #filtr gaussa
    kernel_size = 5
    gauss = gaussian_filter(gray,kernel_size)

    # znajdz krawedzie o danym progu
    low_threshold = 25
    high_threshold = 75
    edges = canny(gauss,low_threshold,high_threshold)

    # zdefiniuj region zainteresowania
    test_roi = region_of_interest(edges,vertices)

    #znajdz linie
    lines = hough_lines(test_roi)

    hough_image = np.zeros((*test_roi.shape, 3), dtype=np.uint8)
    # narysuj linie
    draw_lines(hough_image, lines)

    ended = cv2.addWeighted(image, 0.8, hough_image, 1, 0)

    if lines is not None:
        cv2.putText(ended, "Prawy dystans: ", (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ended, str(float("{0:.3f}".format(right_distance))), (200, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ended, "[m]", (280, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ended, "Lewy dystans: ", (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ended, str(float("{0:.3f}".format(left_distance))), (200, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ended, "[m]", (280, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    if pom1 == 1:
        cv2.putText(ended, "Nie mozesz zmienic prawego pasa - ciagla ", (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (255, 255, 255), 2)

    elif pom1 == 2:
        cv2.putText(ended, "Mozesz zmienic prawy pas - przerywana ", (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (255, 255, 255), 2)

    if pom2 == 1:
        cv2.putText(ended, "Nie mozesz zmienic lewego pasa - ciagla ", (0, 80), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (255, 255, 255), 2)

    elif pom2 == 2:
        cv2.putText(ended, "Mozesz zmienic lewy pas - przerywana ", (0, 80), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (255, 255, 255), 2)

    return ended

# reset zmiennych globalnych
reset_globals()
# tutaj sciezka do danego pliku
video = cv2.VideoCapture('asystent_linie/test_input_video.mp4')
# format pliku
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# plik wyjsciowy o zadanej sciezce, formacie, fps  i rozdzielczosci
out = cv2.VideoWriter('test_output_video.avi',fourcc, 20.0, (960,540))

while(True):

    # czytaj klatki wideo
    ret, frame = video.read()
    if ret:
        frames = frames +1
        # proces dla kazdej ramki
        process= process_image(frame)
        # zapisz do pliku wyjsciowego
        out.write(process)
        # pokaz proces na ekranie
        cv2.imshow('video', process)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

