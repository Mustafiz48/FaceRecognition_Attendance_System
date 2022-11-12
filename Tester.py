import warnings

from tkinter import *
from tkinter import messagebox
import numpy as np
import cv2
import pandas as pd
import datetime
import time

warnings.filterwarnings("ignore")


class Tester:
    def __init__(self):
        # self.haarcasecade_path = "Models/haarcascade_frontalface_default.xml"
        self.modelFile = r"Models/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = r"Models/deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        self.trainimage_path = "TrainingImage"
        self.trainimagelabel_path = "TrainingImageLabel/Trainer.yml"
        self.studentdetail_path = ("StudentDetails/studentdetails.csv")
        self.attendance_path = "Attendance/Attendance_sheet.csv"
        self.excel_path = "Attendance/Attendance_sheet.xlsx"
        self.name = None
        self.id_ = None

    def test(self):
        future = time.time() + 10
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read(self.trainimagelabel_path)
            except:
                e = "Model not found,please train model"
                print(e)
            df = pd.read_csv(self.studentdetail_path)
            cam = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            while True:
                ___, im = cam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                h, w = im.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
                self.net.setInput(blob)
                faces = self.net.forward()
                for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence > 0.5:
                        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x1, y1) = box.astype("int")
                        temp_id, distance = recognizer.predict(gray[y: y1, x: x1])

                        if distance < 55:
                            self.id_ = temp_id
                            try:
                                self.name = df.loc[df["Enrollment"] == self.id_]["Name"].values[0]
                            except Exception as e:
                                print("No entry found at stusdentdetails.csv", e)
                            text = str(self.id_) + "-" + self.name

                            # print(f"Id: {self.id_}, Name: {self.name}")

                            cv2.rectangle(im, (x, y), (x1, y1), (0, 260, 0), 4)
                            cv2.putText(im, str(text), (x1, y), font, 1, (255, 255, 0,), 4)

                        else:
                            Id = "Unknown"
                            text = str(Id)
                            cv2.rectangle(im, (x, y), (x1, y1), (0, 25, 255), 7)
                            cv2.putText(im, str(text), (x1, y), font, 1, (0, 25, 255), 4)
                        break
                cv2.imshow("Filling Attendance...", im)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                if time.time() >= future:
                    cv2.destroyAllWindows()
                    break

        except:
            print("No Face found for attendance")
            cv2.destroyAllWindows()

        if time.time() >= future and self.id_ and self.name:
            try:
                date = datetime.datetime.now().strftime("%d-%m-%Y")  # date object
            except Exception as e:
                print("Error in date time. exception: ", e)
            try:
                attendance_sheet = pd.read_csv(self.attendance_path)
            except Exception as e:
                print("Error in reading csv file. Exception: ", e)
            attendance_sheet = attendance_sheet.drop_duplicates(["Enrollment"], keep="first")

            if str(date) not in attendance_sheet.columns:
                try:
                    attendance_sheet.insert(len(attendance_sheet.columns) - 1, str(date), 0)
                except Exception as e:
                    print("Couldn't insert new date column. Exception: ", e)
            if self.id_ not in attendance_sheet['Enrollment'].values:
                new_row = [self.id_, self.name]
                attendance_sheet = attendance_sheet.append(
                    pd.Series(new_row, index=attendance_sheet.columns[:len(new_row)]),
                    ignore_index=True)
            attendance_sheet.fillna(0, inplace=True)

            indx = attendance_sheet.loc[attendance_sheet["Enrollment"] == self.id_].index
            try:
                attendance_sheet[str(date)][indx] = 1

                attendance_sheet.loc[:, 'Attendance'] = 0

                for i in range(len(attendance_sheet)):
                    attendance_sheet["Attendance"].iloc[i] = str(
                        int(round(attendance_sheet.iloc[i, 2:-1].mean() * 100))) + '%'
                try:
                    attendance_sheet.to_csv(self.attendance_path, index=False)
                    attendance_sheet.to_excel(self.attendance_path[:-3] + ".xlsx", index=False)
                except Exception as e:
                    print("Couldn't save/update attendance sheet.\n", e)

                root = Tk()
                root.withdraw()
                messagebox.showinfo(title="Success",
                                    message=f"Name: {self.name} \n"
                                            f"Id:{self.id_} \n"
                                            f"Attendance taken successfully", )
                root.update()
                root.destroy()
            except Exception as e:
                print(e)
