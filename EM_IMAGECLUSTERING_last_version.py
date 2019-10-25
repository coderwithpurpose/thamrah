
# coding: utf-8

# In[232]:


import numpy as np
import scipy as sp
from scipy import ndimage
import time
import copy
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

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
        self.original_mu = np.mean(self.image_data)
#         self.image_data = self.image_data - np.mean(self.image_data)
#         self.image_data = self.image_data/np.std(self.image_data, axis=0)
        cov = np.cov(self.image_data, rowvar=False)
        a_mat = np.linalg.cholesky(cov)
        a_mat_inv = np.linalg.inv(a_mat)
        self.image_data = np.matmul(self.image_data, a_mat_inv)
        self.segments = segments
        self.pi = np.random.uniform(0, 1, self.segments)
        # self.mu = self.image_data[np.random.choice(self.image_data.shape[0], self.segments, replace=False), :]
        self.mu = kmeans(self.image_data, 10)[0]
        print(self.mu)
        self.past_likelihood, self.curr_likelihood = -np.inf, 0

        tolerance = 1e-2
        iterations = 0
        # We implement the EM algorithm until the likelihood converges
        while True:
            loop_start = time.time()
            self.e_step()
            self.compute_likelihood()

            # Break if difference in current and past likelihood drops below tolerance
            print("The diff currently: ", (self.curr_likelihood - self.past_likelihood))
#             print("number of iterations reached is ", iterations)
            if np.abs(self.curr_likelihood - self.past_likelihood) < tolerance:
#               print(time.time()-loop_start)
                break
            # if iterations>=30:
            #     break
            iterations+=1
            self.m_step()
            print(self.mu)
            print(abs(np.sum(self.mu - self.past_mu)))
            if abs(np.sum(self.mu - self.past_mu)) < tolerance:
                break
        print(np.matmul(self.mu, a_mat))
        self.mu = np.matmul(self.mu, a_mat)


    def e_step(self):
        """
        Performs the e_step
        :param:
        :return:
        """

        w_inter_mat = np.zeros((self.image_data.shape[0], self.segments))
        w = np.zeros(w_inter_mat.shape)

        for j in np.arange(self.segments):
            x_hat = self.image_data - self.mu[j] 
            ### shouldn't be over sqrt(2pi) /// i added the traqnspose part 
            w_inter_mat[:, j] = self.pi[j]*np.exp(-0.5 * np.sum((x_hat)*(x_hat), axis=1))
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
            likelihood += (-0.5*np.sum((x_hat)* x_hat, axis=1) + np.log(self.pi[j]))*self.w[:, j]

        self.past_likelihood = copy.deepcopy(self.curr_likelihood)
        self.curr_likelihood = np.sum(likelihood)

    def m_step(self):
        """
        Performs the m_step
        :return:
        """

        self.past_mu = copy.deepcopy(self.mu)
        self.past_pi = copy.deepcopy(self.pi)
        for j in np.arange(self.segments):
            numer = np.sum(self.image_data*np.reshape(self.w[:, j], (self.w.shape[0], 1)), axis=0) 
#             print(np.shape(np.reshape(self.w[:, j], (self.w.shape[0], 1))))
#             numer = np.sum((np.reshape(self.w[:, j], (self.w.shape[0], 1)) * self.image_data),axis=0)
#             print(np.sum(numer,axis=0))
            denom = np.sum(self.w[:, j])
            self.mu[j] = numer/denom
            self.pi[j] = denom/self.image_data.shape[0]
    

filename = "RobertMixed03.jpg"
em_model = em_segmentation(filename)

import operator
def segment_image(image,model):
        """
        Segments the image data.
        :return:
        """ 
        ### find the closest center for each pixel
        index = np.zeros(len(model.w))
        for idx in range(len(model.w)):
            index[idx],_ = max(enumerate(model.w[idx]), key=operator.itemgetter(1))
            # now assign for the value with the closet index 
            image[idx] = model.mu[int(index[idx])]
        print(np.unique(index)) 
        ## output the new image in (307200,3) shape
        return image
# print(np.shape(em_model.image_data)) 
img =  ndimage.imread(filename, mode='RGB')
# print(np.shape(img))  
image = np.zeros(np.shape(em_model.image_data))
# print(np.shape(image))
new_image = segment_image(image , em_model) 
clustered_image = new_image.reshape(np.shape(img))
print(np.shape(clustered_image))
plt.figure(1)
plt.imshow(clustered_image.astype(int))
plt.show(1)

