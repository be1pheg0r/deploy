import os
import cv2 as cv
import numpy as np
import tensorflow
from tensorflow.keras.utils import image_dataset_from_directory


class Preprocessing:

    def __init__(self, input_folder: str, output_folder: str) -> None:
        '''

        Parameters
        ----------
        input_folder
        output_folder
        '''
        self._input_folder = input_folder
        self._output_folder = output_folder

    def _preprocess(self, ignore_count: bool = 0) -> None:
        '''

        Parameters
        ----------
        ignore_count

        Returns
        -------

        '''

        def _count_cond(path: str) -> int:
            '''

            Parameters
            ----------
            path

            Returns
            -------

            '''
            num_files = 0
            for root, dirs, files in os.walk(path):
                num_files += len(files)
            return num_files

        if not ignore_count:
            if _count_cond(self._input_folder) == _count_cond(self._output_folder):
                return

        def _img_preprocessing(path: str) -> np.ndarray:
            '''

            Parameters
            ----------
            path

            Returns
            -------

            '''
            rgba_image = cv.imread(path, cv.IMREAD_UNCHANGED)
            alpha_channel = rgba_image[:, :, 3]
            letter_mask = (alpha_channel != 0)
            white_background = 255 * np.ones_like(alpha_channel)
            rgb_image = cv.cvtColor(rgba_image, cv.COLOR_BGRA2BGR)
            rgb_image[:, :, 0] = cv.bitwise_or(cv.bitwise_and(alpha_channel, rgb_image[:, :, 0]), white_background)
            rgb_image[:, :, 1] = cv.bitwise_or(cv.bitwise_and(alpha_channel, rgb_image[:, :, 1]), white_background)
            rgb_image[:, :, 2] = cv.bitwise_or(cv.bitwise_and(alpha_channel, rgb_image[:, :, 2]), white_background)
            rgb_image[letter_mask] = (0, 0, 0)
            img = rgb_image
            img = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=(255, 255, 255))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
            img_erode = cv.erode(thresh, np.ones((22, 22), np.uint8), iterations=1)
            contours, hierarchy = cv.findContours(img_erode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            output = img.copy()
            cv.drawContours(output, contours[1:], -1, (255, 0, 0), 1, cv.LINE_AA, hierarchy[1:], 3)
            for idx, contour in enumerate(contours):
                (x, y, w, h) = cv.boundingRect(contour)
                if hierarchy[0][idx][3] == 0:
                    cv.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                    letter_crop = gray[y:y + h, x:x + w]
                    size_max = max(w, h)
                    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                    if w > h:
                        y_pos = size_max // 2 - h // 2
                        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                    elif w < h:
                        x_pos = size_max // 2 - w // 2
                        letter_square[0:h, x_pos:x_pos + w] = letter_crop
                    else:
                        letter_square = letter_crop
                    img = cv.resize(letter_square, (28, 28), interpolation=cv.INTER_AREA)
                    letter = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]

                    return letter

        if not os.path.exists(self._output_folder):
            os.makedirs(self._output_folder)

        for letter_folder in os.listdir(self._input_folder):
            letter_folder_path = os.path.join(self._input_folder, letter_folder)
            if os.path.isdir(letter_folder_path):
                output_letter_folder = os.path.join(self._output_folder, letter_folder)
                if not os.path.exists(output_letter_folder):
                    os.makedirs(output_letter_folder)
                for image_name in os.listdir(letter_folder_path):
                    path = os.path.join(letter_folder_path, image_name)
                    img = _img_preprocessing(path)
                    output_image_path = os.path.join(output_letter_folder, image_name)
                    cv.imwrite(output_image_path, img)

    def dataset(self, ignore_autotune: bool = 0) -> (tensorflow.data.Dataset, tensorflow.data.Dataset):
        '''

        Parameters
        ----------
        self
        ignore_autotune

        Returns
        -------

        '''
        number_of_classes = len(os.listdir(self._output_folder))
        train_data, val_data = image_dataset_from_directory(self._output_folder,
                                                            label_mode='int',
                                                            class_names=[str(x) for x in range(0, number_of_classes)],
                                                            color_mode='grayscale',
                                                            batch_size=32,
                                                            image_size=(28, 28),
                                                            subset='both',
                                                            validation_split=0.25,
                                                            seed=19)
        if not ignore_autotune:
            AUTOTUNE = tensorflow.data.AUTOTUNE
            train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
            val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

        return train_data, val_data