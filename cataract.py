import os.path

import numpy as np
from tensorflow.keras.models import load_model
from input import Input


class Cataract:

    def __init__(self, model_language_mode: str = 'both_languages') -> None:
        '''

        Parameters
        ----------
        model_language_mode: ['both_languages' | 'cyrillic' |  'latin']
        '''
        self._lang_code = None
        # 'both_languages' = 0
        # 'cyrillic' = 1
        # 'latin' = 2
        languages = {
            'both_languages': 'all_symbols_both_languages_model.h5',
            'cyrillic': 'all_symbols_cyrillic_model.h5',
            'latin': 'all_symbols_latin_model.h5'
        }
        if model_language_mode in languages.keys():
            self._lang_code = list(languages.keys()).index(model_language_mode)
            self._model = load_model(os.path.join('models/', languages[model_language_mode]))
        else:
            raise ValueError('incorrect language mode')

    def __str__(self):

        message = ('''                    ----------------
                        Cataract
                     Version: 1.0
                     Supported languages:
                     - Russian
                     - English
                     - Numbers
                     ----------------    
                   ''')
        return message

    def _predict(self, input: np.ndarray) -> int:
        '''

        Parameters
        ----------
        input

        Returns
        -------

        '''
        return np.argmax(self._model.predict(np.reshape(input, (1, 28, 28, 1))))

    def img_to_str(self, path: str) -> str:
        '''

        Parameters
        ----------
        path

        Returns
        -------

        '''
        answer = []
        letters = Input.get_letters(path)
        alph = {
            10: 'а',  # 43 английская
            19: 'и',
            20: 'й',
            21: 'к',  # 53 английская
            22: 'л',
            23: 'м',  # 55 английская
            24: 'н',  # 50 английская
            25: 'о',  # 57 английская
            26: 'п',
            27: 'р',  # 58 английская
            28: 'с',  # 45 английская
            11: 'б',
            29: 'т',  # 62 английская
            30: 'у',  # 67 английская
            31: 'ф',
            32: 'х',  # 66 английская
            33: 'ц',
            34: 'ч',
            35: 'ш',
            36: 'щ',
            37: 'ъ',
            38: 'ы',
            12: 'в',  # 44 английская
            39: 'ь',
            40: 'э',
            41: 'ю',
            42: 'я',
            13: 'г',
            14: 'д',  # 46 английская
            15: 'е',  # 47 английская
            16: 'ё',
            17: 'ж',
            18: 'з',
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            43: 'a',  # 10 русская
            52: 'j',
            53: 'k',  # 21 русская
            54: 'l',
            55: 'm',  # 23 русская
            56: 'n',
            57: 'o',  # 25 русская
            58: 'p',  # 27 русская
            59: 'q',
            60: 'r',
            61: 's',
            44: 'b',  # 12 русская
            62: 't',  # 29 русская
            63: 'u',
            64: 'v',
            65: 'w',
            66: 'x',  # 32 русская
            67: 'y',  # 30 русская
            68: 'z',
            45: 'c',  # 28 русская
            46: 'd',  # 14 русская
            47: 'e',  # 15 русская
            48: 'f',
            49: 'g',
            50: 'h',  # 24 русская
            51: 'i'
        }
        if self._lang_code == 0:

            trouble_pairs = [[10, 43], [21, 53], [23, 55], [24, 50], [25, 57], [27, 58], [28, 45],
                             [29, 62], [30, 67], [32, 66], [12, 44], [14, 46], [15, 47]]
            trouble = [j for i in trouble_pairs for j in i]
            numbs = [i for i in range(10)]
            rus = [i for i in range(10, 43)]
            en = [i for i in range(43, 69)]
            structure = [0, 0, 0]
            for letter in letters:
                prediction = self._predict(letter)
                answer.append(prediction)
                res = (int((prediction in numbs)) * 0 + int((prediction in rus)) * 1 + int((prediction in en)) * 2)
                structure[res] += 1
            language = [rus, numbs, en][structure.index(max(structure))]
            for l in range(len(answer)):
                if answer[l] in trouble:
                    for pair in trouble_pairs:
                        if answer[l] in pair:
                            if answer[l] not in language:
                                answer[l] = pair[int(not (pair.index(answer[l])))]
                answer[l] = alph[answer[l]]
        else:
            for letter in letters:
                prediction = self._predict(letter)
                answer.append(alph[prediction])
        return ''.join(answer)

