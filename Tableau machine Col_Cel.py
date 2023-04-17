import tensorflow as tf
import cv2
import numpy as np
import imutils

# Créer un modèle de détection de tableaux et de colonnes
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compiler le modèle avec une fonction de perte binaire croisée et un optimiseur Adam
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Charger les images et les étiquettes d'entraînement et de validation
train_data = []  # liste des images d'entraînement
train_labels = []  # liste des étiquettes correspondantes (0 pour un tableau, 1 pour une colonne)
val_data = []  # liste des images de validation
val_labels = []  # liste des étiquettes correspondantes

# Ajouter des images et des étiquettes à partir de votre jeu de données d'entraînement et de validation
# ...

# Convertir les listes en tableaux numpy
train_data = np.array(train_data)
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

# Entraîner le modèle sur les données d'entraînement
history = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

# Évaluer la précision du modèle sur les données de validation
loss, accuracy = model.evaluate(val_data, val_labels)

# Charger l'image et la redimensionner
img = cv2.imread('nom_de_votre_image.jpg')
img = cv2.resize(img, (800, 800))

# Convertir l'image en échelle de gris et la flouter
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Appliquer un seuillage adaptatif pour détecter les contours
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Trouver les contours et trier par aire décroissante
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Boucler sur les contours et trouver le plus grand rectangle qui est un multiple de 10 pour être un tableau
tableau_contour = None
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        ( # Vérifier si le rectangle est un multiple de 10 et si sa hauteur et sa largeur sont d'au moins 100 pixels
        if (approx[1][0][0] - approx[0][0][0]) % 10 == 0 and (approx[2][0][0] - approx[1][0][0]) % 10 == 0 and \
                (approx[2][0][1] - approx[0][0][1]) % 10 == 0 and (approx[3][0][1] - approx[1][0][1]) % 10 == 0 and \
                (approx[2][0][0] - approx[0][0][0]) >= 100 and (approx[2][0][1] - approx[0][0][1]) >= 100:
            tableau_contour = approx
            break

# Extraire le tableau de l'image et le redimensionner à une taille fixe
tableau_img = four_point_transform(img, tableau_contour.reshape(4, 2))
tableau_img = cv2.resize(tableau_img, (800, 800))

# Appliquer le modèle de détection de colonnes et de cellules au tableau
tableau_gray = cv2.cvtColor(tableau_img, cv2.COLOR_BGR2GRAY)
tableau_gray = cv2.GaussianBlur(tableau_gray, (5, 5), 0)
tableau_thresh = cv2.adaptiveThreshold(tableau_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Trouver les contours des colonnes
colonnes_contours = cv2.findContours(tableau_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
colonnes_contours = imutils.grab_contours(colonnes_contours)
colonnes_contours = sorted(colonnes_contours, key=cv2.boundingRect, reverse=False)

# Boucler sur les contours des colonnes et identifier les cellules de chaque colonne
for colonne_contour in colonnes_contours:
    # Extraire la colonne de l'image et la redimensionner à une taille fixe
    colonne_img = tableau_img[:, colonne_contour[0][0]:colonne_contour[1][0]]
    colonne_img = cv2.resize(colonne_img, (200, 800))
    
    # Appliquer le modèle de détection de cellules à la colonne
    colonne_gray = cv2.cvtColor(colonne_img, cv2.COLOR_BGR2GRAY)
    colonne_gray = cv2.GaussianBlur(colonne_gray, (5, 5), 0)
    colonne_thresh = cv2.adaptiveThreshold(colonne_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Trouver les contours des cellules
    cellules_contours = cv2.findContours(colonne_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cellules_contours = imutils.grab_contours(cellules_contours)
    cellules_contours = sorted(cellules_contours, key=cv2.boundingRect, reverse=False)
    
    # Boucler sur les contours des cellules et afficher les rectangles autour des cellules
    for cellule_contour in cellules_contours:
        # Dessiner un rectangle autour de la cellule
        cv2.rectangle(colonne_img, cellule_contour[ 0][0][0], cellule_contour[0][0][1]), (cellule_contour[1][0][0], cellule_contour[1][0][1]), (0, 255, 0), 2)

    # Afficher la colonne avec les rectangles autour des cellules détectées
    cv2.imshow("Colonne", colonne_img)
    cv2.waitKey(0)

# Afficher le tableau avec les colonnes et les cellules détectées
cv2.imshow("Tableau", tableau_img)
cv2.waitKey(0)