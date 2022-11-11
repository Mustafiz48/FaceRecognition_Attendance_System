import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

import tkinter as tk
from tkinter import simpledialog


class Register:
    def __init__(self):
        self.haarcasecade_path = "haarcascade_frontalface_default.xml"
        self.haarcasecade_path_2 = "haarcascade_frontalface_alt.xml"
        self.trainimage_path = "TrainingImage"
        self.trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
        self.student_detail_path = "StudentDetails/studentdetails.csv"
        self.attendance_sheet = "Attendance/Attendance_sheet.csv"

    # take Image of user
    def take_image(self):
        root = tk.Tk()
        root.withdraw()
        root.geometry("700x350")

        # the input dialog
        user_name = simpledialog.askstring(title="Name", prompt="Please enter your Name")
        enrollment_id = simpledialog.askstring(title="Id", prompt="Please enter your Id")
        root.update()
        root.destroy()
        if not user_name or not enrollment_id:
            print("Please enter valid username, id and try again")
            return
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(self.haarcasecade_path)
            detector_2 = cv2.CascadeClassifier(self.haarcasecade_path_2)
            sample_num = 0
            directory = enrollment_id + "_" + user_name
            path = os.path.join(self.trainimage_path, directory)
            os.mkdir(path)
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num = sample_num + 1
                    cv2.imwrite(f"{path}" +"\\"+ user_name + "_" + enrollment_id + "_" + str(sample_num) + ".jpg",
                                gray[y: y + h, x: x + w], )
                    cv2.imshow("Frame", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif sample_num > 50:
                    break
            cam.release()
            cv2.destroyAllWindows()
            row = [enrollment_id, user_name]
            with open(self.student_detail_path, "a+", ) as csvFile:
                writer = csv.writer(csvFile, delimiter=",")
                writer.writerow(row)
                csvFile.close()

            with open(self.attendance_sheet, "a+", ) as csvFile:
                writer = csv.writer(csvFile, delimiter=",")
                writer.writerow(row)
                csvFile.close()

        except FileExistsError as F:
            print("Student Data already exists")
