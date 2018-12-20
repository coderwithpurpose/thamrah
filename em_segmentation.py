import numpy as np
import scipy as sp
from scipy import ndimage
import time
import copy

class em_segmentation:
    """
    Performs segmentation on images using the specified number of segments
    """

    def __init__(self, filename, segments=10):

        # Set up the parameters for the gaussian assuming that the covariance is identity
        self.image_data = ndimage.imread(filename, mode='RGB')
        ## making the image a list of rgb pixels -> [r1g1b1],..... and so on
        self.image_data = self.image_data.reshape(self.image_data.shape[0]*self.image_data.shape[1], self.image_data.shape[2])

        # Center the data and make it unit variance
        self.image_data = self.image_data - np.mean(self.image_data)
        self.image_data = self.image_data/np.std(self.image_data, axis=0)

        self.segments = segments
        self.pi = np.random.uniform(0, 1, self.segments)
        self.mu = self.image_data[np.random.choice(self.image_data.shape[0], self.segments, replace=False), :]
        self.past_likelihood, self.curr_likelihood = -np.inf, 0

        tolerance = 1e-2
        # We implement the EM algorithm until the likelihood converges
        while True:
            loop_start = time.time()
            self.e_step()
            self.compute_likelihood()

            # Break if difference in current and past likelihood drops below tolerance
            print("The diff currently: ", (self.curr_likelihood - self.past_likelihood))
            if np.abs(self.curr_likelihood - self.past_likelihood) < tolerance:
                break

            self.m_step()

    def e_step(self):
        """
        Performs the e_step
        :param:
        :return:
        """

        w_inter_mat = np.zeros((self.image_data.shape[0], self.segments))
        w = np.zeros(w_inter_mat.shape)
        # for i in np.arange(self.image_data.shape[0]):
        #     for j in np.arange(self.segments):
        #         x_hat = self.image_data[i, :] - self.mu[j]
        #         w_inter_mat[i, j] = self.pi[j]*np.exp(-0.5 * np.matmul(x_hat.T, x_hat))
        #     w[i, :] = w_inter_mat[i, :]/np.sum(w_inter_mat[i, :])

        for j in np.arange(self.segments):
            x_hat = self.image_data - self.mu[j]
            w_inter_mat[:, j] = self.pi[j]*np.exp(-0.5 * np.sum(x_hat*x_hat, axis=1))
        for i in np.arange(self.image_data.shape[0]):
            w[i, :] = w_inter_mat[i, :]/np.sum(w_inter_mat[i, :])
        self.w = w

    def compute_likelihood(self):
        """
        Computes the likelihood after the e_step
        :return:
        """

        likelihood = np.zeros((self.image_data.shape[0]))

        for j in np.arange(self.segments):
            x_hat = self.image_data - self.mu[j]
            likelihood += (-0.5*np.sum(x_hat * x_hat, axis=1) + np.log(self.pi[j]))*self.w[:, j]

        self.past_likelihood = self.curr_likelihood
        self.curr_likelihood = np.sum(likelihood)

    def m_step(self):
        """
        Performs the m_step
        :return:
        """
        #
        # for j in np.arange(self.segments):
        #     numer, denom = 0, 0
        #     for i in np.arange(self.image_data.shape[0]):
        #         numer += self.image_data[i, :]*self.w[i, j]
        #         denom += self.w[i, j]
        #     self.mu[j] = numer/denom
        #     self.pi[j] = denom/self.image_data.shape[0]


        past_mu = copy.deepcopy(self.mu)
        past_pi = copy.deepcopy(self.pi)
        # print(past_mu)
        for j in np.arange(self.segments):
            numer = np.sum(self.image_data*np.reshape(self.w[:, j], (self.w.shape[0], 1)), axis=0)
            denom = np.sum(self.w[:, j])
            self.mu[j] = numer/denom
            self.pi[j] = denom/self.image_data.shape[0]

        # print(self.mu)
        # print(past_mu)
        # print("MU difference: ", np.sum(self.mu - past_mu))
        # print("PI difference: ", np.sum(self.pi - past_pi))



    def segment_image(self):
        """
        Segments the image data.
        :return:
        """
        pass


filename = "/Users/soorajkumar/Desktop/CS 498 AML/MP8/sample_image_1.jpg"
em_model = em_segmentation(filename)


