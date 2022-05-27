import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model as kload_model
from keras.preprocessing import image as kimage
from PIL import ImageGrab
from typing import NoReturn, Any, List, Dict, Tuple, Text, Callable


class Classifiers():
    def __init__(self, name: str) -> NoReturn:
        """class untuk mendapatkan classifiers dalam bentuk objek

        params
        ---
        name: str
            nama berdasarkan nama file classifiers tanpa extension
            path: ..Lib\site-packages\cv2\data
        returns
        ---

        """
        self.__name = name

    def create(self) -> cv2.CascadeClassifier:
        """membuat objek CascadeClassifier

        params
        ---

        returns
        ---
        CascadeClassifier
            objek CascadeClassifier

        """
        retval = None
        cv2_path = os.path.dirname(cv2.__file__)
        data_path = os.path.join(cv2_path, "data")

        for f in os.listdir(data_path):
            fname, ext = os.path.splitext(f)
            if ext == ".xml" and fname == self.__name:
                data_file = os.path.join(data_path, f).replace("\\", "/")
                retval = cv2.CascadeClassifier(data_file)

        return retval


class Shape():
    RECTANGLE = 1

    def __init__(self, shape_id: int = 1, color: Tuple = (255, 0, 0),
                 text_color: Tuple = (0, 255, 0), thickness=2) -> NoReturn:
        """class untuk prepare membuat bentuk

        params
        ---
        shape_id: int
            id bentuk/geometry dari opencv
        color: tuple
            code warna untuk batas bentuk/geometry
        text_color: tuple
            code warna untuk text
        tickness: int
            ketebalan text/batas bentuk/geometry

        returns
        ---

        """
        self.__color = color
        self.__text_color = text_color
        self.__shape_id = shape_id
        self.__thickness = thickness

    def create(self, image, text: str = "", positions: List = [], sizes: List = []) -> NoReturn:
        """untuk mulai membuat bentuk

        params
        ---
        image: object
            object image dari opencv
        text: str
            teks yang akan muncul di antara bentuk
        positions: list
            posisi dalam x dan y
        sizes: list
            ukuran dalam width dan height

        returns
        ---

        """
        size = (positions[0] + sizes[0],
                positions[1] + sizes[1])
        text_position = (positions[0], positions[1] - 10)

        if self.__shape_id == self.RECTANGLE:
            cv2.rectangle(image, tuple(positions), size,
                          self.__color, self.__thickness)
            cv2.putText(image, text,
                        text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, self.__text_color, self.__thickness)


class CamCapture():
    def __init__(self, classifier: Classifiers, shape: Shape) -> NoReturn:
        self.__capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.__classifier = classifier
        self.__shape = shape

    @ property
    def capture(self) -> Any:
        return self.__capture

    @ property
    def shape(self) -> Any:
        return self.__shape

    def _generate(self) -> List:
        retval = []
        is_read, img = self.__capture.read()
        img = cv2.flip(img, 1)
        if is_read:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            csf = self.__classifier.create()
            if csf:
                faces = csf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(
                    30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                retval = [img, gray, faces]
        return retval

    def _stop(self) -> NoReturn:
        self.__capture.release()
        cv2.destroyAllWindows()

    def create(self) -> NoReturn:
        while True:
            img, _, faces = self._generate()
            for (x, y, w, h) in faces:
                if not bool(self.loop_modify()):
                    self.__shape.create(img, [x, y], [w, h])
                    cv2.imshow('Cam Capture', img)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
        self._stop()

    def loop_modify(self) -> bool:
        return False


class EmotionCapture(CamCapture):
    MARAH = 0
    JIJIK = 1
    TAKUT = 2
    SENANG = 3
    BIASA = 4
    SEDIH = 5
    TERKEJUT = 6

    def __init__(self, classifier: Classifiers, shape: Shape) -> NoReturn:
        super().__init__(classifier, shape)
        self.__kclassifier = kload_model(r"model\model.h5")

    @ classmethod
    def get_text(cls, emotion_id: int) -> Text:
        retval = None
        for k, v in cls.__dict__.items():
            if "__" not in k and type(v) is int:
                if v == emotion_id:
                    retval = k

        return retval

    def loop_modify(self) -> bool:
        img, gray, faces = self._generate()

        for (x, y, w, h) in faces:
            roi_gray = gray[y: y + h, x: x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = kimage.image_utils.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.__kclassifier.predict(roi)[0]
                text = self.get_text(prediction.argmax())
                self.shape.create(img, text, [x, y], [w, h])

        cv2.imshow("Emotion Capture", img)
        return True

    # def create(self) -> NoReturn:
    #     while True:
    #         img, gray, faces = self._generate()

    #         for (x, y, w, h) in faces:
    #             roi_gray = gray[y: y + h, x: x + w]
    #             roi_gray = cv2.resize(roi_gray, (48, 48),
    #                                   interpolation=cv2.INTER_AREA)

    #             if np.sum([roi_gray]) != 0:
    #                 roi = roi_gray.astype('float') / 255.0
    #                 roi = kimage.image_utils.img_to_array(roi)
    #                 roi = np.expand_dims(roi, axis=0)

    #                 prediction = self.__kclassifier.predict(roi)[0]
    #                 text = self.get_text(prediction.argmax())
    #                 self.shape.create(img, text, [x, y], [w, h])

    #         cv2.imshow("Emotion Capture", img)
    #         if cv2.waitKey(30) & 0xff == ord('q'):
    #             break

    #     self._stop()


if __name__ == "__main__":
    # cap = CamCapture(Classifiers("haarcascade_frontalface_alt2"), Shape())
    # cap.create()
    emoCap = EmotionCapture(Classifiers(
        "haarcascade_frontalface_default"), Shape())
    emoCap.create()
