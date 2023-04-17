import tensorflow as tf
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET

# Définir les classes d'objet à détecter
class_names = ['tableau']

# Chargement des annotations XML pour chaque image
def load_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_names:
            continue

        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])

    return np.array(boxes)

# Chargement des images
def load_images(img_file):
    return cv2.imread(img_file)

# Fonction pour extraire les caractéristiques d'une image avec un modèle pré-entrainé
def extract_features(model, img):
    # Redimensionner l'image à la taille attendue par le modèle
    img = cv2.resize(img, (224, 224))

    # Normaliser les valeurs de pixel de l'image
    img = img.astype('float32') / 255.0

    # Ajouter une dimension pour le batch
    img = np.expand_dims(img, axis=0)

    # Extraire les caractéristiques de l'image
    features = model.predict(img)

    return features

# Chargement du modèle de détection d'objet pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Création du modèle de détection d'objets personnalisé
inputs = tf.keras.Input(shape=(7, 7, 1280))
x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
x = tf.keras.layers.Dense(4, activation='linear')(x)
model = tf.keras.Model(inputs=inputs, outputs=x)

# Compilation du modèle
model.compile(loss='mse', optimizer='adam')

# Entraînement du modèle
images_dir = 'chemin_vers_votre_répertoire_images'
annotations_dir = 'chemin_vers_votre_répertoire_annotations'

images = []
annotations = []
for file in os.listdir(images_dir):
    if file.endswith('.jpg'):
        images.append(load_images(os.path.join(images_dir, file)))
        annotations.append(load_annotations(os.path.join(annotations_dir, file[:-4] + '.xml')))

images = np.array(images)
annotations = np.array(annotations)

model.fit(extract_features(model, images), annotations, epochs=10, batch_size=32)
