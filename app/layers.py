print('print python')

# custom L1 distance layer module

# import dependencies

import tensorflow as tf
from keras.layers import Layer

# custome L1 distance Layer from jupyter
# this need to loasd the custome model
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self,inputEmbedding,validationEmbedding):
        return tf.math.abs(inputEmbedding-validationEmbedding)
    



