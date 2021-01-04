from enum import Enum
import random
import math
import collections
from abc import ABC, abstractmethod
import numpy as np


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2],
              [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
              [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1],
              [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
              [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4],
              [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5], [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2],
              [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
              [5, 3.2, 1.2, 0.2], [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
              [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2], [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2],
              [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5],
              [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3],
              [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1], [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4],
              [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4], [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5],
              [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7],
              [6, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3],
              [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2], [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2],
              [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3],
              [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8], [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9], [6.8, 3, 5.5, 2.1],
              [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2],
              [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2],
              [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8],
              [6.4, 2.8, 5.6, 2.1], [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3],
              [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5], [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9],
              [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, 30)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12], [0.13],
             [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25], [0.26],
             [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39],
             [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51], [0.52],
             [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64], [0.65],
             [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78],
             [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9], [0.91],
             [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03], [1.04],
             [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16], [1.17],
             [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29], [1.3],
             [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42], [1.43],
             [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55], [1.56],
             [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919], [0.129634142619695],
             [0.139543114644236], [0.149438132473599], [0.159318206614246], [0.169182349066996], [0.179029573425824],
             [0.188858894976501], [0.198669330795061], [0.2084598998461], [0.218229623080869], [0.227977523535188],
             [0.237702626427135], [0.247403959254523], [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134], [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814], [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983],
             [0.380188415123161], [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211],
             [0.425939465066], [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483],
             [0.470625888171158], [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969], [0.548023936791874],
             [0.556361022912784], [0.564642473395035], [0.572867460100481], [0.581035160537305], [0.58914475794227],
             [0.597195441362392], [0.60518640573604], [0.613116851973434], [0.62098598703656], [0.628793024018469],
             [0.636537182221968], [0.644217687237691], [0.651833771021537], [0.659384671971473], [0.666869635003698],
             [0.674287911628145], [0.681638760023334], [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523], [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859], [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363], [0.857298989188603],
             [0.862404227243338], [0.867423225594017], [0.872355482344986], [0.877200504274682], [0.881957806884948],
             [0.886626914449487], [0.891207360061435], [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883], [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802],
             [0.928368967249167], [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692], [0.958015860289225],
             [0.960835064206073], [0.963558185417193], [0.966184951612734], [0.968715100118265], [0.971148377921045],
             [0.973484541695319], [0.975723357826659], [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846], [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588], [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054], [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479], [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, 10)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    examples = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(examples, labels, 100)
    network.train(data, 10001, order=NNData.Order.RANDOM)


def run_tests():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.add_hidden_layer(4)
    network.layers.reset_cur()
    while True:
        print(network.layers.current.get_layer_info())
        if not network.layers.iterate():
            break
    network.layers.reset_cur()
    network.layers.iterate()
    network.layers.remove_hidden_layer()
    network.layers.reset_cur()
    while True:
        print(network.layers.current.get_layer_info())
        if not network.layers.iterate():
            break


def main():
    run_tests()
    run_iris()
    run_sin()
    run_XOR()


class DataMismatchError(Exception):
    """ Exception raised if set sizes are mismatched"""
    pass


class NNData:
    """
    NNData trains and tests data
    """
    def __init__(self, x=None, y=None, percentage=100):
        if x is None:
            x = []
        if y is None:
            y = []
        self.x = None
        self.y = None
        self.train_percentage = NNData.percentage_limiter(percentage)
        self.train_indices = None
        self.train_pool = None
        self.test_indices = None
        self.test_pool = None
        self.load_data(x, y)

    class Order(Enum):
        """
        Class Order creates Enum literals that denotes whether the order of the pools is
        randomized or sequential
        """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """
        Class Set creates Enum literals that denote whether a set is to be trained or tested
        """
        TRAIN = 0
        TEST = 1

    def load_data(self, x, y):
        """
        Checks the lengths of lists x and y to see if they're equal, raises DataMismatchError
        if not, assigns values to self.x and self.y, and calls split_set. Returns nothing
        """
        if not len(x) == len(y):
            raise DataMismatchError
        self.x = x
        self.y = y
        self.split_set()

    def split_set(self, new_train_percentage=None):
        """
        Calculate the size of test and training sets using train_percentage and the size
        of the data set, and fill them randomly. Call prime_data
        """
        if new_train_percentage is not None:
            self.train_percentage = self.percentage_limiter(new_train_percentage)
        train_size = math.floor((self.train_percentage / 100) * len(self.x))
        self.train_indices = random.sample(range(0, len(self.x)), train_size)
        self.test_indices = list(set(range(0, len(self.x))) - set(self.train_indices))
        self.prime_data()

    def prime_data(self, my_set=None, order=None):
        """
        Copy train and/or test indices into train and/or test pools based on assigned
        my_set. Shuffle or copy as is based on Order.
        """
        if order is None:
            order = NNData.Order.SEQUENTIAL

        def train_copy():
            train_indices_temp = list(self.train_indices)
            if order is NNData.Order.RANDOM:
                random.shuffle(train_indices_temp)
            self.train_pool = collections.deque(train_indices_temp)
            return self.train_pool

        def test_copy():
            test_indices_temp = list(self.test_indices)
            if order is NNData.Order.RANDOM:
                random.shuffle(test_indices_temp)
            self.test_pool = collections.deque(test_indices_temp)
            return self.test_pool

        if my_set is None:
            train_copy()
            test_copy()
        elif my_set is NNData.Set.TRAIN:
            train_copy()
        elif my_set is NNData.Set.TEST:
            test_copy()

    def empty_pool(self, my_set=None):
        """
        Check if train or test pools are empty, return True or False
        """
        if my_set is None:
            my_set = NNData.Set.TRAIN
        if my_set is NNData.Set.TRAIN:
            if len(self.train_pool) == 0:
                return True
            else:
                return False
        else:
            if len(self.test_pool) == 0:
                return True
            else:
                return False

    def get_number_samples(self, my_set=None):
        """
        Checks and returns number of samples in my_set
        """
        if my_set is None:
            return len(self.x)
        elif my_set is NNData.Set.TRAIN:
            return len(self.train_indices)
        elif my_set is NNData.Set.TEST:
            return len(self.test_indices)

    def get_one_item(self, my_set=None):
        """
        Take leftmost index from pool associated with my_set and return
        [x,y], where x and y are corresponding labels and examples from
        data set
        """
        if my_set is None:
            my_set = NNData.Set.TRAIN
        if my_set is NNData.Set.TRAIN:
            while self.train_pool:
                index = self.train_pool.popleft()
                return [self.x[index], self.y[index]]
        elif my_set is NNData.Set.TEST:
            while self.test_pool:
                index = self.test_pool.popleft()
                return [self.x[index], self.y[index]]

    @staticmethod
    def percentage_limiter(percentage):
        """
        Modifies and returns percentage value if it is not between 0 and 100 inclusive
        """
        if percentage < 0:
            percentage = 0
        elif percentage > 100:
            percentage = 100
        return percentage


class MultiLinkNode(ABC):
    """
    Abstract class inherited by class Neurode()
    """
    def __init__(self):
        self.input_nodes = collections.OrderedDict([])
        self.output_nodes = collections.OrderedDict([])
        self.number_inputs = 0
        self.number_outputs = 0
        self.reporting_inputs = 0
        self.reporting_outputs = 0
        self.compare_inputs_full = 0
        self.compare_outputs_full = 0

    def __str__(self):
        ret_str = "-->Node" + str(id(self)) + "\n"
        ret_str = ret_str + "   Input Nodes:\n"
        for key in self.input_nodes:
            ret_str = ret_str + "  " + str(id(key)) + "\n"
        ret_str = ret_str + "   Output Nodes\n"
        for key in self.output_nodes:
            ret_str = ret_str + "  " + str(id(key)) + "\n"
        return ret_str

    def clear_and_add_input_nodes(self, nodes):
        """
        Function takes a list of nodes as a parameter, clears
        self.input_nodes, and adds the list of nodes to the dictionary
        """
        self.input_nodes.clear()
        for node in nodes:
            self.input_nodes[node] = None
            self.process_new_input_node(node)
        self.number_inputs = len(self.input_nodes)
        self.compare_inputs_full = 2 ** self.number_inputs - 1

    def clear_and_add_output_nodes(self, nodes):
        """
        Function takes a list of nodes as a parameter, clears
        self.output_nodes, and adds the list of nodes to the dictionary
        """
        self.output_nodes.clear()
        for node in nodes:
            self.output_nodes[node] = None
            self.process_new_output_node(node)
        self.number_outputs = len(self.output_nodes)
        self.compare_outputs_full = 2 ** self.number_outputs - 1

    @abstractmethod
    def process_new_input_node(self, node):
        pass

    @abstractmethod
    def process_new_output_node(self, node):
        pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Neurode(MultiLinkNode):
    """
    This is MultiLinkNode's child class, it works on
    individual nodes in our neural network
    """
    def __init__(self, my_type):
        super().__init__()
        self.value = 0
        self.my_type = my_type

    def process_new_input_node(self, node):
        """
        Parameter node is the key for dictionary self.input_nodes,
        this function assigns a random number between 0 and 1 to
        the value of given key.
        """
        node_weight = random.random()
        self.input_nodes[node] = node_weight

    def process_new_output_node(self, node):
        """
        Doesn't need to do anything
        """
        pass

    def get_value(self):
        """
        Accessor function that returns self.value
        """
        return self.value

    def get_type(self):
        """
        Accessor function that returns self.my_type
        """
        return self.my_type


class FFNeurode(Neurode):
    """
    The feed forward Neurode subclass
    """
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def activate_sigmoid(value):
        """
        Process value using sigmoid function and return result
        """
        return 1 / (1 + np.exp(-value))

    def receive_input(self, from_node=None, input_value=0):
        """
        Accept data inputs and process their values
        call fire() when data is available for all inputs
        to let it know that the neurode has a value ready
        """
        if self.my_type is LayerType.INPUT:
            self.value = input_value
            for node in self.output_nodes:
                node.receive_input(self)
        elif self.register_input(from_node):
            self.fire()

    def register_input(self, from_node):
        """
        Updates number of reporting inputs, checks to see if all inputs are reporting,
        and returns a boolean depending on whether or not all inputs are reporting,
        resets self.reporting_inputs to 0 if True
        """
        node_index = list(self.input_nodes.keys()).index(from_node)
        self.reporting_inputs += 2 ** node_index
        if self.reporting_inputs == self.compare_inputs_full:
            self.reporting_inputs = 0
            return True
        else:
            return False

    def fire(self):
        """
        Determines a value for the node by calculating a weighted sum based on the values
        and weights of the inputs, calls self.activate_sigmoid and then notifies
        all output neurodes that a value is ready
        """
        weighted_sum = 0
        for node in self.input_nodes:
            weighted_sum += (self.input_nodes[node] * node.value)
        self.value = self.activate_sigmoid(weighted_sum)
        for next_layer_node in self.output_nodes:
            next_layer_node.receive_input(self)


class BPNeurode(Neurode):
    """
    The Back-Propagation Neurode subclass
    """
    def __init__(self, my_type):
        super().__init__(my_type)
        self.delta = 0
        self.learning_rate = 0.05

    @staticmethod
    def sigmoid_derivative(value):
        return value * (1 - value)

    def receive_back_input(self, from_node, expected=0):
        """
        Collects signals from nearby neurons and decides when to fire
        """
        if self.my_type is LayerType.OUTPUT:
            self.calculate_delta(expected)
            self.back_fire()
        elif self.register_back_input(from_node):
            self.calculate_delta(expected)
            self.back_fire()
            self.update_weights()

    def register_back_input(self, from_node):
        """
        Updates number of reporting inputs, checks to see if all inputs are reporting,
        and returns a boolean depending on whether or not all inputs are reporting,
        resets self.reporting_inputs to 0 if True. If the node is an output node,
        this will return True.
        """
        if self.my_type is LayerType.OUTPUT:
            return True
        elif self.my_type is LayerType.HIDDEN:
            node_index = list(self.output_nodes.keys()).index(from_node)
            self.reporting_outputs += 2 ** node_index
            if self.reporting_outputs == self.compare_outputs_full:
                self.reporting_outputs = 0
                return True
            else:
                return False

    def calculate_delta(self, expected=None):
        """
        Updates self.data.
        """
        if self.my_type is LayerType.OUTPUT:
            output_delta = (expected - self.value) * self.sigmoid_derivative(self.value)
            self.delta = output_delta
        elif self.my_type is LayerType.HIDDEN:
            weighted_deltas = 0
            for node in self.output_nodes:
                weight = node.get_weight_for_input_node(self)
                weighted_deltas += weight * node.delta
            hidden_delta = weighted_deltas * self.sigmoid_derivative(self.value)
            self.delta = hidden_delta

    def update_weights(self):
        """
        Updates incoming weights as described in modules
        """
        for key, node_data in self.output_nodes.items():
            adjustment = key.get_learning_rate() * key.get_delta() * self.value
            key.adjust_input_node(self, adjustment)

    def back_fire(self):
        """
        Calls receive_back_input on each of the neurode's input nodes
        """
        for node in self.input_nodes:
            node.receive_back_input(self)

    def get_learning_rate(self):
        """
        Accessor for self.learning_rate
        """
        return self.learning_rate

    def get_delta(self):
        """
        Accessor for self.delta
        """
        return self.delta

    def get_weight_for_input_node(self, from_node):
        """
        Accessor for the input node weight for from_node
        """
        weight = self.input_nodes[from_node]
        return weight

    def adjust_input_node(self, node, value):
        """
        Mutator that adds value to input_nodes[node]
        """
        self.input_nodes[node] += value


class FFBPNeurode(FFNeurode, BPNeurode):
    pass

class DLLNode:
    """
    The DLLNode class represents individual node objects in our
    Doubly Linked List
    """
    def __init__(self):
        self.next = None
        self.prev = None

    def set_next(self, next_node):
        """
        Sets next_node as the node following current node
        """
        self.next = next_node

    def set_prev(self, prev_node):
        """
        Sets prev_node as the node before current node
        """
        self.prev = prev_node

    def get_next(self):
        """
        Accessor for next node
        """
        return self.next

    def get_prev(self):
        """
        Accessor for previous node
        """
        return self.prev


class DoublyLinkedList:
    """
    The DoublyLinkedList class represents the linked list we are using in our
    neural network
    """
    def __init__(self):
        self.head = None
        self.current = None
        self.tail = None

    def reset_cur(self):
        """
        Resets the current element of the Linked List to the head
        """
        self.current = self.head
        return self.current

    def iterate(self):
        """
        Iterates through the linked list in the forward direction. Returns
        the next element
        """
        if self.current is not None:
            self.current = self.current.get_next()
        return self.current

    def rev_iterate(self):
        """
        Iterates through the linked list in the reverse direction. Returns
        the previous element
        """
        if self.current is not None:
            self.current = self.current.get_prev()
        return self.current

    def add_to_head(self, new_node):
        """
        Adds element new_node to the beginning of the list
        """
        if isinstance(new_node, DLLNode):
            new_node.set_next(self.head)
            if self.head is not None:
                self.head.set_prev(new_node)
            self.head = new_node
            if self.head.get_next() is None:
                self.tail = self.head

    def remove_from_head(self):
        """
        Removes first element of linked list
        """
        ret_node = self.head
        if ret_node is not None:
            self.head = ret_node.get_next()
            ret_node.set_next(None)
            if self.head:
                self.head.set_prev(None)
        if self.current is ret_node:
            self.current = None
        if self.tail is ret_node:
            self.tail = None
        return ret_node

    def insert_after_cur(self, new_node):
        """
        Inserts element new_node after self.current in linked list
        """
        if isinstance(new_node, DLLNode) and self.current:
            if self.current == self.tail:
                self.current.set_next(new_node)
                new_node.set_prev(self.current)
                self.tail = new_node
            else:
                new_node.set_next(self.current.get_next())
                self.current.get_next().set_prev(new_node)
                self.current.set_next(new_node)
                new_node.set_prev(self.current)
            return True
        else:
            return False

    def remove_after_cur(self):
        """
        Removes element following current element in linked list
        """
        if not self.current or not self.current.get_next():
            return False
        elif self.current.get_next() is self.tail:
            self.current.set_next(None)
            self.tail = self.current
        else:
            self.current.set_next(self.current.get_next().get_next())
            self.current.get_next().set_prev(self.current)

    def is_empty(self):
        """
        Checks to see if linked list is empty
        """
        return self.head is None

    def __str__(self):
        SEPARATOR = " -> "
        LEN_SEPARATOR = len(SEPARATOR)
        if self.is_empty():
            return "\n[ empty list ]\n"
        ret_str = "\n[START LIST]: "
        p = self.head
        while p is not None:
            ret_str += (str(p) + SEPARATOR)
            p = p.get_next()
        ret_str = ret_str[:-LEN_SEPARATOR] + "  [END LIST]\n"
        return ret_str


class NodePositionError(Exception):
    """
    Exception raised if client tries to remove an input or output layer
    """
    pass


class Layer(DLLNode):
    """
    Child class of DLLNode that gives each layer its own nodes for our neural network
    """
    def __init__(self, num_neurodes=5, my_type=LayerType.HIDDEN):
        super().__init__()
        self.my_type = my_type
        self.neurodes = []
        for node in range(num_neurodes):
            self.add_neurode()

    def add_neurode(self):
        """
        We will use it to add neurodes of type FFBPNeurode to
        the layer.
        """
        self.neurodes.append(FFBPNeurode(self.my_type))

    def get_my_neurodes(self):
        """
        Accessor for self.neurodes
        """
        return self.neurodes

    def get_my_type(self):
        """
        Accessor for self.my_type
        """
        return self.my_type

    def get_layer_info(self):
        """
        Accessor that returns a tuple of (layer type, number of neurodes)
        """
        return self.my_type, len(self.neurodes)


class LayerList(DoublyLinkedList):
    """
    Child class of DoublyLinkedList that adds node functionality within each layer of the list
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        input_layer = Layer(num_inputs, LayerType.INPUT)
        output_layer = Layer(num_outputs, LayerType.OUTPUT)
        self.add_to_head(input_layer)
        self.reset_cur()
        self.insert_after_cur(output_layer)

    def insert_after_cur(self, new_layer):
        """
        Private class that inserts a new layer to our linked list, and makes the connections
        between the nodes in each layer
        """
        if self.current.get_my_type() is LayerType.OUTPUT:
            raise NodePositionError
        for neurode in new_layer.get_my_neurodes():
            neurode.clear_and_add_input_nodes(self.current.get_my_neurodes())
        for neurode in self.current.get_my_neurodes():
            neurode.clear_and_add_output_nodes(new_layer.get_my_neurodes())
        if new_layer.get_my_type() is LayerType.HIDDEN:
            for neurode in new_layer.get_my_neurodes():
                neurode.clear_and_add_output_nodes(self.current.get_next().get_my_neurodes())
            for neurode in self.current.next.get_my_neurodes():
                neurode.clear_and_add_input_nodes(new_layer.get_my_neurodes())
        super().insert_after_cur(new_layer)

    def remove_after_cur(self):
        """
        Private class that de-links the neurodes in the layer we want to remove, and appropriately
        re links the remaining layers' neurodes
        """
        for node in self.current.get_next().neurodes:
            node.input_nodes = None
            node.output_nodes = None
        for node in self.current.neurodes:
            node.clear_and_add_output_nodes(self.current.get_next().get_next().neurodes)
        for node in self.current.get_next().get_next().neurodes:
            node.clear_and_add_input_nodes(self.current.neurodes)
        super().remove_after_cur()

    def insert_hidden_layer(self, num_neurodes):
        """
        Uses private method __insert_after_cur() to add a new hidden Layer containing num_neurodes
        to our LayerList after self.current
        """
        if self.current == self.tail:
            raise NodePositionError
        else:
            self.insert_after_cur(Layer(num_neurodes))

    def remove_hidden_layer(self):
        """
        Uses private method __remove_after_cur() to remove the hidden layer after self.current
        """
        if not self.current or not self.current.get_next() or self.current.get_next() == self.tail:
            raise NodePositionError
        else:
            self.remove_after_cur()

    def get_input_nodes(self):
        """
        Accessor for the list of neurodes at the head
        """
        return self.head.get_my_neurodes()

    def get_output_nodes(self):
        """
        Accessor for list of neurodes at the tail
        """
        return self.tail.get_my_neurodes()


class FFBPNetwork:
    """
    class FFBPNetwork has a LayerList and has NNData, and uses these to train and test neural networks
    """
    class EmptyLayerException(Exception):
        """
        Raises error if user tries to add a layer with less than one neurode
        """
        pass

    class EmptySetException(Exception):
        """
        Raises error if NNData object has no examples loaded
        """
        pass

    def __init__(self, num_inputs, num_ouputs):
        self.layers = LayerList(num_inputs, num_ouputs)

    def add_hidden_layer(self, num_neurodes=5):
        """
        Adds a new hidden layer to our LayerList, raises EmptyLayerException if num_neurodes < 1
        """
        if num_neurodes < 1:
            raise self.EmptyLayerException
        else:
            self.layers.insert_hidden_layer(num_neurodes)

    def remove_hidden_layer(self):
        """
        Removes a hidden layer from the neural network by calling LayerList function on
        self.layers
        """
        self.layers.remove_hidden_layer()

    def iterate(self):
        """
        Iterates through LayerList and returns to client
        """
        return self.layers.iterate()

    def rev_iterate(self):
        """
        Reverse iterates through LayerList and returns to client,
        uses DoublyLinkedList's defined function
        """
        return self.layers.rev_iterate()

    def reset_cur(self):
        """
        Resets current in LayerList to the inputs layer
        """
        return self.layers.reset_cur()

    def get_layer_info(self):
        """
        Returns layer information for current layer
        """
        return self.layers.current.get_layer_info()

    def train(self, data_set: NNData, epochs=1000, verbosity=0, order=NNData.Order.SEQUENTIAL):
        """
        Trains given data_set, running through the set for the given
        number of epochs, and prints the sample's errors based on given verbosity.
        Returns the RMSE of the dataset
        """
        if not data_set:
            raise self.EmptySetException
        else:
            rmse_error = 0
            for training_round in range(1, epochs + 1):
                data_set.prime_data(NNData.Set.TRAIN, order)
                rooted_errors = 0
                while data_set.empty_pool() is False:
                    squared_errors = 0
                    self.reset_cur()
                    [inputs, expected] = data_set.get_one_item()
                    for (value, node) in zip(inputs, self.layers.get_input_nodes()):
                        node.receive_input(None, value)
                    for (value, node) in zip(expected, self.layers.get_output_nodes()):
                        squared_errors += (value - node.value) ** 2
                        node.receive_back_input(None, value)
                    if verbosity > 1 and training_round % 1000 == 0:
                        predicted_list = []
                        for node in self.layers.get_output_nodes():
                            predicted_list.append(node.value)
                        print("Predicted: %s  Expected: %s" % (str(predicted_list),str(expected)))
                    rooted_errors += math.sqrt(squared_errors)
                rmse_error = math.sqrt(rooted_errors / data_set.get_number_samples())
                if verbosity > 0:
                    if training_round % 100 == 0:
                        print("Epoch: %i RMSE: %f" % (training_round, rmse_error))
            return rmse_error

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """
        Runs through the data set in the forward direction once and prints the inputs, expected,
        and predicted values for all of the samples
        """
        if not data_set:
            raise self.EmptySetException
        else:
            squared_errors = 0
            rooted_errors = 0
            data_set.prime_data(NNData.Set.TEST, order)
            while data_set.empty_pool(NNData.Set.TEST) is False:
                predicted = []
                self.reset_cur()
                [inputs, expected] = data_set.get_one_item(NNData.Set.TEST)
                for (value, node) in zip(inputs, self.layers.get_input_nodes()):
                    node.receive_input(None, value)
                for (value, node) in zip(expected, self.layers.get_output_nodes()):
                    squared_errors += (value - node.value) ** 2
                    predicted.append(node.value)
                print("Input: %s, Expected Value: %s, Predicted Value: %s" % (str(inputs), str(expected),
                                                                              str(predicted)))
                rooted_errors += math.sqrt(squared_errors)
            rmse_error = math.sqrt(rooted_errors / data_set.get_number_samples(NNData.Set.TEST))
            return rmse_error


if __name__ == "__main__":
    main()
