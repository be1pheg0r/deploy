import cv2 as cv
import numpy as np

class Input:

    @staticmethod
    def get_letters(path: str) -> list[np.ndarray]:

        img = cv.imread(path)
        img = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=(255, 255, 255))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
        img_erode = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv.findContours(img_erode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        cv.drawContours(output, contours[1:], -1, (255, 0, 0), 1, cv.LINE_AA, hierarchy[1:], 3)

        letters = []
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
                img = cv.bitwise_not(img)
                img = cv.threshold(img, 77, 255, cv.THRESH_BINARY)[1]
                img = cv.bitwise_not(img)
                letters.append((x, w, img))

        letters.sort(key=lambda x: x[0], reverse=False)

        letters = [i[-1] for i in letters]

        return letters

