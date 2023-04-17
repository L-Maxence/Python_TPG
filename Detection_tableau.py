import cv2
import numpy as np

# Charger l'image dans le programme
img = cv2.imread('nom_de_votre_image.jpg')

# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre de seuillage adaptatif pour segmenter l'image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

# Appliquer une série de dilatations et d'érosions pour éliminer le bruit et fermer les contours
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Trouver les contours dans l'image segmentée
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Examiner chaque contour pour déterminer s'il correspond à un tableau sans bordure
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
    
    # Vérifier si le contour a quatre côtés
    if len(approx) == 4:
        # Vérifier si les angles sont approximativement droits
        angles = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i+1)%4][0]
            pt3 = approx[(i+2)%4][0]
            v1 = np.array([pt2[0]-pt1[0], pt2[1]-pt1[1]])
            v2 = np.array([pt3[0]-pt2[0], pt3[1]-pt2[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cos_angle)
            angles.append(angle)
        if max(angles) < np.pi/4:
            # Dessiner le contour correspondant sur l'image originale
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

# Afficher l'image avec les tableaux identifiés
cv2.imshow('Tableaux détectés', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
