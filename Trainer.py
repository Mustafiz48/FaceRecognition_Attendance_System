import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image


class Trainer:
    def __init__(self):
        self.haarcasecade_path = "haarcascade_frontalface_default.xml"
        self.trainimage_path = "TrainingImage"
        self.trainimagelabel_path = "TrainingImageLabel/Trainer.yml"

    # Train Image
    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self.haarcasecade_path)
        faces, id_ = self.get_images_and_labels(self.trainimage_path)
        recognizer.train(faces, np.array(id_))
        with open(self.trainimagelabel_path, 'w+') as f:
            recognizer.save(self.trainimagelabel_path)
        print("Image Trained successfully")

    def get_images_and_labels(self, path):
        # image_path = [os.path.join(path, f) for d in os.listdir(path) for f in d]
        new_dir = [os.path.join(path, d) for d in os.listdir(path)]
        image_path = [
            os.path.join(new_dir[i], f)
            for i in range(len(new_dir))
            for f in os.listdir(new_dir[i])
        ]
        faces = []
        ids = []
        for image_path in image_path:
            pil_image = Image.open(image_path).convert("L")
            image_np = np.array(pil_image, "uint8")
            id_ = int(os.path.split(image_path)[-1].split("_")[1])
            faces.append(image_np)
            ids.append(id_)
        return faces, ids
