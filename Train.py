import tensorflow as tf
import numpy as np
import cv2

# Charger les données d'entraînement à partir d'un fichier .npy
train_data = np.load('train_data.npy')

# Diviser les données en images et en coordonnées en pixels
train_images = train_data[:,0]
train_labels = train_data[:,1]

# Créer un modèle CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compiler le modèle avec une fonction de perte et un optimiseur
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Enregistrer le modèle entraîné
model.save('table_extraction_model')






# Charger l'image à tester
image = cv2.imread('nom_de_votre_image.jpg')

# Prétraitement de l'image
image = preprocess(image)

# Appliquer le modèle pour prédire la position du tableau
predictions = model.predict(np.array([image]))
x1, y1, x2, y2 = predictions[0]

# Afficher les coordonnées prédites sur l'image originale
cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.imshow("Image avec la detection de tableau", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
