import cv2
import numpy as np

# Charger l'image dans le programme
img = cv2.imread('nom_de_votre_image.jpg')

# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre de Canny pour détecter les contours
edges = cv2.Canny(gray, 100, 200)

# Appliquer la transformation de Hough pour détecter les lignes droites
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Examiner les lignes détectées pour trouver des groupes de lignes qui forment des angles droits
quadrilaterals = []
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        for k in range(j+1, len(lines)):
            for l in range(k+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                line3 = lines[k][0]
                line4 = lines[l][0]

                # Vérifier que les quatre lignes forment un quadrilatère
                if abs(np.pi - abs(np.arctan2(line1[3]-line1[1], line1[2]-line1[0]) - np.arctan2(line2[3]-line2[1], line2[2]-line2[0]))) < np.pi/18 and \
                abs(np.pi - abs(np.arctan2(line2[3]-line2[1], line2[2]-line2[0]) - np.arctan2(line3[3]-line3[1], line3[2]-line3[0]))) < np.pi/18 and \
                abs(np.pi - abs(np.arctan2(line3[3]-line3[1], line3[2]-line3[0]) - np.arctan2(line4[3]-line4[1], line4[2]-line4[0]))) < np.pi/18 and \
                abs(np.pi - abs(np.arctan2(line4[3]-line4[1], line4[2]-line4[0]) - np.arctan2(line1[3]-line1[1], line1[2]-line1[0]))) < np.pi/18:

                    # Vérifier que les côtés opposés ont des longueurs similaires
                    if abs(np.linalg.norm(line1[:2] - line2[:2]) - np.linalg.norm(line3[:2] - line4[:2])) / (np.linalg.norm(line1[:2] - line2[:2]) + np.linalg.norm(line3[:2] - line4[:2])) < 0.2 and \
                    abs(np.linalg.norm(line2[:2] - line3[:2]) - np.linalg.norm(line4[:2] - line1[:2])) / (np.linalg.norm(line2[:2] - line3[:2]) + np.linalg.norm(line4[:2] - line1[:2])) < 0.2:

                        # Ajouter le quadrilatère à la liste des tableaux identifiés
                        quadrilaterals.append((line1, line2, line3, line4))

# Dessiner les tableaux identifiés sur l'image
for quadrilateral in quadrilaterals:
    for i in range(4):
       cv2.line(img, tuple(quadrilateral[i][:2]), tuple(quadrilateral[(i+1)%4][:2]), (0, 255, 0), 2)

#Afficher l'image avec les quadrilatères trouvés
cv2.imshow('image avec quadrilateres', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





import cv2
import numpy as np

# Charger l'image en niveaux de gris
img = cv2.imread("image.jpg", 0)

# Appliquer un flou gaussien pour réduire le bruit de l'image
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Appliquer un seuillage adaptatif pour convertir l'image en noir et blanc
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Trouver les contours dans l'image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Parcourir les contours pour trouver le plus grand contour rectangulaire
tableau_contour = None
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
        if len(approx) == 4:
            tableau_contour = approx
            max_area = area

# Dessiner le contour du tableau sur l'image originale
cv2.drawContours(img, [tableau_contour], 0, (0, 255, 0), 2)

# Afficher l'image avec le contour du tableau détecté
cv2.imshow("Tableau detecte", img)
cv2.waitKey(0)


