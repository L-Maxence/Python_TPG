import cv2
import numpy as np
from sklearn.cluster import KMeans

# Chargement de l'image
img = cv2.imread('nom_de_votre_image.jpg')

# Conversion en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre de Canny pour détecter les contours
edges = cv2.Canny(gray, 100, 200)

# Appliquer la transformation de Hough pour détecter les lignes droites
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Création d'un masque pour isoler le tableau
mask = np.zeros_like(gray)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 5)

# Appliquer le masque à l'image originale
masked_image = cv2.bitwise_and(img, img, mask=mask)

# Convertir l'image en niveaux de gris et en noir et blanc
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Appliquer un filtre de flou pour réduire le bruit
blur = cv2.GaussianBlur(threshold, (3, 3), 0)

# Appliquer la détection de contours pour identifier les contours du tableau
contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Identifier les contours du tableau
table_contour = None
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50000:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            table_contour = approx
            max_area = area

# Extraire les coordonnées des coins du tableau
table_coordinates = table_contour.reshape(4, 2)
table_coordinates = table_coordinates[np.argsort(table_coordinates[:, 0])]

if table_coordinates[0][1] > table_coordinates[1][1]:
    table_coordinates[[0, 1]] = table_coordinates[[1, 0]]
if table_coordinates[2][1] < table_coordinates[3][1]:
    table_coordinates[[2, 3]] = table_coordinates[[3, 2]]

# Afficher les coordonnées du tableau
print("Coordonnées du tableau :")
print(table_coordinates)


