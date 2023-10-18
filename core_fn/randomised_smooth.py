import tensorflow as tf
import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from math import ceil

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: tf.keras.Model, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: tf.Tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = np.argmax(counts_selection)
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat]
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: tf.Tensor, n: int, alpha: float, batch_size: int) -> int:
        counts = self._sample_noise(x, n, batch_size)
        top2 = np.argsort(counts)[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        return top2[0]

    def _sample_noise(self, x: tf.Tensor, num: int, batch_size) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=int)
        for _ in range(ceil(num / batch_size)):
        # for i in range(batch_size):
            
            this_batch_size = min(batch_size, num)
            # # this_batch_size = num
            
            num -= this_batch_size

            batch = tf.tile(x, [this_batch_size, 1, 1, 1])
            noise = tf.random.normal(shape=batch.shape, stddev=self.sigma,dtype=tf.float64)
            predictions = tf.argmax(self.base_classifier(batch + noise), axis=-1)
            counts += self._count_arr(predictions.numpy(), self.num_classes)
        return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    # def predict(self, x: tf.Tensor, n: int, alpha: float, batch_size: int) -> int:
    #     counts = self._sample_noise(x, n, batch_size)
    #     top2 = np.argsort(counts,axis = -1)[:,::-1][:,:2]
    #     # count1 = counts[top2[0]]
    #     # count2 = counts[top2[1]]
    #     # if binom_test(count1, count1 + count2, p=0.5) > alpha:
    #     #     return Smooth.ABSTAIN
    #     # else:
    #     count1 = np.max(counts,axis = -1)
    #     return top2[:,0]

    # def _sample_noise(self, x: tf.Tensor, num: int, batch_size) -> np.ndarray:
    #     counts = np.zeros((x.shape[0],self.num_classes), dtype=int)
    #     for _ in range(ceil(num / batch_size)):
    #     # for i in range(batch_size):
            
    #         this_batch_size = min(batch_size, num)
    #         # # this_batch_size = num
            
    #         num -= this_batch_size

    #         batch = tf.tile(x, [this_batch_size, 1, 1, 1])
    #         noise = tf.random.normal(shape=batch.shape, stddev=self.sigma,dtype=tf.float64)
    #         predictions = tf.argmax(self.base_classifier(batch + noise), axis=-1)
    #         counts += self._count_arr(predictions.numpy(), x.shape[0],self.num_classes)
    #     return counts

    # def _count_arr(self, arr: np.ndarray, height:int, length: int) -> np.ndarray:
    #     counts = np.zeros((height, length), dtype=int)
    #     # height is batch_size
    #     arr = arr.reshape((-1, height))
    #     for i in range(arr.shape[0]):
    #         for j in range(arr.shape[1]):
    #             counts[j][arr[i][j]] += 1
    #     return counts