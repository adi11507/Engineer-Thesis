# -------------------------------------------------------------------------------
# Name:        Asystent Ropoznawania Znakow Ograniczenia Predkosci
# Purpose:
#
# Author:      adrian
#
# Created:     19-10-2018
# Copyright:   (c) adrian 2018
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

pixel_in_cm = 0.0254
aktualna_liczba_1 = 0
aktualna_liczba_2 = 0
pomocnicza = 0
aktualna_liczba_trzy_1 = 0
aktualna_liczba_trzy_2 = 0
aktualna_liczba_trzy_3 = 0
circles_count = 0
full_distance = 0
actual_frame = 0
speed = 0
start_frame = 0

# WYKONUJEMY TYLKO RAZ DLA DANEGO KLASYFIKATORA
# Zaladowanie probek z publicznej bazy danych "MNIST Original"
#dane = datasets.fetch_mldata("MNIST Original")
# wczytanie do tablic probek i etykiet
#probki = np.array(dane.data, 'int16')
#etykieta = np.array(dane.target, 'int')

# stworzenie nowej tablicy w ktorej zapisane zostana wyniki funkcji HOG
#lista_funkcja_hog = []
#for probka in probki:
    #funkcja = hog(probka.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
             #visualise=False)
    #lista_funkcja_hog.append(funkcja)
#hog_probki = np.array(lista_funkcja_hog, 'float64')

# tworzenie obiektu
#klasyfikator = LinearSVC()
# LinearSVC(), LogisticRegression(), KNeighborsClassifier
#dopasowanie modelu do danych treningowych
#klasyfikator.fit(hog_probki, etykieta)
# zapisanie klasyfikatora do pliku
#joblib.dump(klasyfikator, "klasyfikator_liczb.pkl", compress=3)

def kontrast_img(image):
    # zmiana kontrastu zdjecia
    # zamień na YCrcB
    img_to_YCrcB = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_to_YCrcB)
    channels[0] = cv2.equalizeHist(channels[0])
    img_merged = cv2.merge(channels)
    img_to_return = cv2.cvtColor(img_merged, cv2.COLOR_YCrCb2BGR)

    return img_to_return


def gauss_laplace(image):
    # zastosowanie filtru gaussowskiego
    gauss_laplace_image = cv2.GaussianBlur(image, (3, 3), 0)
    # konwersja na skale szarosci
    gray = cv2.cvtColor(gauss_laplace_image, cv2.COLOR_BGR2GRAY)
    # filtr laplace
    gauss_laplace_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    gauss_laplace_image = cv2.convertScaleAbs(gauss_laplace_image)

    return gauss_laplace_image


def binaryzacja(image):
    # progowanie pikseli
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]

    return thresh


def preprocess_image(image):
    # wykorzystanie wszystkich funkcji do wstepnego przetwarzania obrazu
    image = kontrast_img(image)
    image = gauss_laplace(image)
    image = binaryzacja(image)
    return image

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

def region_zainteresowania(image, vertices):
    # definicja regionu zainteresowania
    mask = np.zeros_like(image)

    # stworzenie pustego obrazu
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # wypelnienie ze wzgledu na wierzcholki
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# funkcja znajdowania znaku
def find_sign(image,detect,circles):

    global full_distance
    image_process = preprocess_image(image)
    removed_image = usun_male_komponenty(image_process, 200)

    # znajdz kola na obrazie w danym zakresie promienia
    circles = cv2.HoughCircles(removed_image, cv2.HOUGH_GRADIENT, 1.3, 200, param1=100, param2=50, minRadius=20,
                               maxRadius=30)

    # stworzenie nowych oddzielnych macierzy i przypisanie do nich parametrow
    # (x,y) oraz r znalezionego kola
    x_p = []
    y_p = []
    r_p = []

    if circles is None:
        return image, detect, circles

    else:
        # konwersja parametrow na calkowite
        circles = np.round(circles[0, :]).astype("int")

        # zapisanie znalezionych parametrow kol do macierzy
        for (x, y, r) in circles:
            x_p.append(x)
            y_p.append(y)
            full_distance = y
            r_p.append(r)

            #zaznaczenie znaku na obrazie
            cv2.rectangle(detect, (x - r - 10, y - r - 5), (x + r + 10, y + r + 5), (0, 255, 0), 2)

        x_new = max(x_p)
        y_new = max(y_p)
        r_new = max(r_p)

        # zdefiniowanie wierzcholkow regionu zainteresowania
        vertices = np.array([[(x_new - r_new - 5, y_new - r_new - 5), (x_new + r_new +5, y_new - r_new - 5),
                              (x_new + r_new + 5, y_new + r_new + 5),
                              (x_new - r_new - 5, y_new + r_new + 5)]], dtype=np.int32)

        # wyciecie zdjecia z regionu zainteresowania
        cropped_image = region_zainteresowania(image, vertices)
        cropped_image_sign = cropped_image[(y_new - r_new + 10 ):(y_new + r_new -10),
                        (x_new - r_new +5):(x_new + r_new -5 )]


        return cropped_image_sign, detect, circles

def detektor_liczb(image, liczba): #identyfikacja liczb

    image[np.where((image > [0,0,55]).all(axis=2))] = [255,255,255]

    # zaladowanie klasyfikatora (tutaj knn)
    # dostepne na plycie, SVC oraz Logistic Regression
    klasyfikator = joblib.load("klasyfikator_liczb_knn.pkl")
    # konwersja na skale szarosci oraz filtr gaussowski
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # progowanie obrazu
    ret, image_prog = cv2.threshold(image_gray, 170 , 255, cv2.THRESH_BINARY_INV)

    image_prog = usun_male_komponenty(image_prog,30)

    # znajdz kontury na obrazie
    _, ctrs, hierarchy = cv2.findContours(image_prog.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # zbuduj prostokaty znalezionych kontur
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # dla regionów oblicz HOG oraz predykcje
    for rect in rects:

        new_image = image_prog[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

        # prostokatny region dookola liczby
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = image[pt1:pt1 + leng, pt2:pt2 + leng]

        # powrot zdjecia do wielkosci bazowej
        roi = cv2.resize(new_image, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3,3))
        deskew_image = deskew(roi,roi.shape[1],roi.shape[0])


        # kalkulacja HOG, identyfikacja cyfr, zapisanie ich do macierzy
        roi_hog_fd = hog(deskew_image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = klasyfikator.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(image, str(int(nbr[0])), (rect[0]-5, rect[1]+30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        liczba.append(nbr)


    return image, liczba

def process(image):

    global aktualna_liczba_1
    global aktualna_liczba_2
    global aktualna_liczba_trzy_1
    global aktualna_liczba_trzy_2
    global aktualna_liczba_trzy_3
    global pomocnicza
    global actual_frame
    global circles_count
    global start_frame
    global speed
    global full_distance

    liczby = []
    circles = []
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    detect = np.copy(image)

    cv2.putText(detect, "Obowiazujaca predkosc: ", (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    cropped_image_sign, detect, circles = find_sign(image, detect, circles)

    width = cropped_image_sign.shape[1]
    height = cropped_image_sign[0]

    if (width,height) < (35,35):
       return detect

    # dodatkowe narzedzia obliczania predkosci
    #if circles is not None:
        #start_frame = 1
    #else:
        #start_frame = 0

    #if actual_frame > 0 and circles is None:
        #speed = (full_distance * pixel_in_cm) / (actual_frame / fps)

    #if speed > 0:
        #cv2.putText(detect, "Twoja predkosc: ", (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        #cv2.putText(detect, str(int(speed)), (300, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    if circles is not None:
        znak, liczby = detektor_liczb(cropped_image_sign, liczby)
        print(liczby)
        if len(liczby) == 2:
            liczby.sort()
            aktualna_liczba_1 = liczby[1]
            print(aktualna_liczba_1)
            aktualna_liczba_2 = liczby[0]
            print(aktualna_liczba_2)

        if len(liczby) == 3:
            pomocnicza = 1
            aktualna_liczba_trzy_1 = liczby[2]
            aktualna_liczba_trzy_2 = liczby[1]
            aktualna_liczba_trzy_3 = liczby[0]

    if (aktualna_liczba_1) > 0:
        cv2.putText(detect, str(int(aktualna_liczba_1)), (300, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(detect, str(int(aktualna_liczba_2)), (315, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        #if (aktualna_liczba_1 * 10 + aktualna_liczba_2) < speed:
            #cv2.putText(detect, "Zwolnij! ", (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    else:
        if pomocnicza == 1:
            cv2.putText(detect, str(int(aktualna_liczba_trzy_1)), (300, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(detect, str(int(aktualna_liczba_trzy_2)), (315, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(detect, str(int(aktualna_liczba_trzy_3)), (330, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        #if (aktualna_liczba_trzy_1 * 100 + aktualna_liczba_trzy_2 * 10 + aktualna_liczba_trzy_3) < speed:
            #cv2.putText(detect, "Zwolnij! ", (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)


    return detect

def deskew(img,width,height):
    # wyrownanie zdjecia rotacji itd..
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # jesli mniejsze zwroc
        return img.copy()
    # oblicz momenty obrazu
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.6*height*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return img

def reset_global():
    global aktualna_liczba_1
    global aktualna_liczba_2
    global aktualna_liczba_trzy_1
    global aktualna_liczba_trzy_2
    global aktualna_liczba_trzy_3
    global pomocnicza
    aktualna_liczba_1 = 0
    aktualna_liczba_2 = 0
    pomocnicza = 0
    aktualna_liczba_trzy_1 = 0
    aktualna_liczba_trzy_2 = 0
    aktualna_liczba_trzy_3 = 0
    start_frame = 0
    actual_frame = 0
    speed = 0

#reset zmiennych globalnych
reset_global()
# tutaj wpisz sciezke pliku video
video = cv2.VideoCapture('asystent_znaki/test_video_sign.mp4')
# nadaj format pliku video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# do czego, gdzie, ile fps, rozdzielczosc zapisac wyjsciowy obraz
out = cv2.VideoWriter('asystent_znaki/test_1.avi',fourcc, 30.0, (960,540))
# zworc ilosc fps video
fps = video.get(cv2.CAP_PROP_FPS)

# petla dopoki prawdziwa
while(True):

    # czytaj video ramka po ramce
    ret, frame = video.read()
    if ret:

        # funkcja procesowa dla kazdej ramki
        process_image = process(frame)

        # zliczanie ramek do obliczenia predkosci
        if start_frame == 1:
            actual_frame = actual_frame + 1
        else:
            actual_frame = 0

        # zapisanie kazdej z ramek do kolejnego pliku
        out.write(process_image)
        # pokaz wideo wyjsciowe
        cv2.imshow('video', process_image)

        # wcisnij q aby wyjsc
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()