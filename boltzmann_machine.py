import numpy as np
#from keras.datasets import mnist
#from mnist import MNIST
import mnist as mndata
import random
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import mean_squared_error 

HIDDEN = 1
VISIBLE = 0

class boltzmann_machine:
    """
    This class denoises images from the MNIST dataset using a boltzmann-machine. We output the original, the noisy
    and the denoised images.
    """

    def __init__(self):
        self.c = 0.2
        self.c2 = 0.2
        self.get_and_process_MNIST_data()
        self.set_up_MRF()
        self.denoise_images()
        pass

    def get_and_process_MNIST_data(self):
        """
        This function simply retrieves the MNIST trainings samples and labels and stores them in the class.
        It then binarizes the data and 'noises' the data by randomly flipping 2% of the pixels per image.
        :return: None
        """

        #mndata = MNIST() 
        #self.train_images, self.train_labels = mndata.load_training() 
        self.train_images, self.train_labels = np.reshape(mndata.train_images(),(60000,784)), mndata.train_labels()
        self.train_images, self.train_labels = self.train_images[:500], self.train_labels[:500]  
        print(np.shape(self.train_images)) 
        print(np.shape(self.train_labels)) 
        ## labeling the pixels back 
        self.train_images, self.train_labels = np.array([[1 if p > 0.5 else -1 for p in i] for i in self.train_images]), np.array(self.train_labels)
        
        ### i need to change the below code so it iterate through the matrix properly 
        #self.train_images, self.train_labels = np.array([[1 if p > 0.5 else -1 for p in i] for i in self.train_images), np.array(self.train_labels)
        side_length = int(np.sqrt(self.train_images.shape[1]))
        self.orig_train_images = copy.deepcopy(self.train_images.reshape((self.train_images.shape[0], side_length, side_length)))
        self.noisy_train_images = np.zeros((self.train_images.shape[0], side_length, side_length))
        for im in np.arange(self.train_images.shape[0]):
            random_inds = random.sample(range(1, self.train_images.shape[1]), int(0.02 * self.train_images.shape[1]))
            self.train_images[im, random_inds] = np.where(self.train_images[im, random_inds] == -1, 1, -1)
            self.noisy_train_images[im, :, :] = self.train_images[im, :].reshape(side_length, side_length)
        self.side_length = side_length

    def set_up_MRF(self):
        """
        This function sets up the MRFs that we are to use to denoise images. It also sets up a PI array for each image
        :return: MRF 3D array
        """

        self.mrf = np.ones((self.train_images.shape[0], self.side_length, self.side_length, 2))
        self.mrf[:, :, :, VISIBLE] = self.noisy_train_images
        self.pi = np.ones((self.train_images.shape[0], self.side_length, self.side_length)) * 0.5

    def denoise_images(self):
        """
        This function denoises images by testing convergence of the pi matrix per image.
        :return:
        """

        tol = 1e-5
        accuracies = []
        self.denoised_images = np.zeros(self.noisy_train_images.shape)
        for im in np.arange(self.noisy_train_images.shape[0]):
            print(im)
            new_pi_array = np.zeros((self.side_length, self.side_length))
            while True:
                for y in np.arange(self.side_length):
                    for x in np.arange(self.side_length):
                        first_term, second_term = self.get_terms((x, y), im)
                        # print("Old pi", self.pi[im, x, y])
                        new_pi = first_term/(first_term + second_term)
                        # print("New pi", new_pi)
                        new_pi_array[x, y] = new_pi
                if abs(np.sum(new_pi_array - self.pi[im, :, :])) < tol:
                    break
                else:
                    self.pi[im, :, :] = new_pi_array
            pi_list = self.pi[im, :, :].flatten().tolist()
            dn_im = np.array([1 if x >= 0.5 else -1 for x in pi_list]).reshape((self.side_length, self.side_length))
            self.denoised_images[im, :, :] = dn_im
            # plt.figure(1)
            # plt.imshow(self.noisy_train_images[im, :, :])
            # plt.show(1)
            # plt.figure(1)
            # plt.imshow(dn_im)
            # plt.show(1)

            # print("Accuracy: ", np.sum(self.orig_train_images[im, :, :] == self.denoised_images[im, :, :])/(self.side_length*self.side_length))
            accuracies += [np.sum(self.orig_train_images[im, :, :] == self.denoised_images[im, :, :])/(self.side_length*self.side_length)]
        self.accuracies = accuracies
        best_index = np.argmax(accuracies)
        print("The best image is: ", best_index)
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.imshow(self.orig_train_images[best_index, :, :])
        plt.subplot(3, 1, 2)
        plt.imshow(self.noisy_train_images[best_index, :, :])
        plt.subplot(3, 1, 3)
        plt.imshow(self.denoised_images[best_index, :, :])
        plt.show(1)

        worst_index = np.argmin(accuracies)
        print("The worst image is: ", worst_index)
        plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.imshow(self.orig_train_images[worst_index, :, :])
        plt.subplot(3, 1, 2)
        plt.imshow(self.noisy_train_images[worst_index, :, :])
        plt.subplot(3, 1, 3)
        plt.imshow(self.denoised_images[worst_index, :, :])
        plt.show(2)

    def get_terms(self, coords, im):
        """
        This function returns the terms that are used to calculate the new PIs.
        :param: coord - A tuple of the coordinate of the pixel in the image
        :return: tuple of terms
        """

        x, y = coords

        visible_n = self.mrf[im, x, y, VISIBLE]*(self.c2)
        visible_n_2 = self.mrf[im, x, y, VISIBLE]*(-self.c2)

        left_n, left_n_2 = 0, 0
        if x - 1 >= 0:
            # left_n = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x-1, y, HIDDEN]) else (self.c*(2*self.pi[im, x-1, y] - 1)))
            # left_n_2 = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x-1, y, HIDDEN]) else (-self.c*(2*self.pi[im, x-1, y] - 1)))
            left_n = ((self.c * (2 * self.pi[im, x - 1, y] - 1)))
            left_n_2 = ((-self.c * (2 * self.pi[im, x - 1, y] - 1)))

        right_n, right_n_2 = 0, 0
        if x + 1 < self.noisy_train_images.shape[1]:
            # right_n = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x+1, y, HIDDEN]) else (self.c*(2*self.pi[im, x+1, y] - 1)))
            # right_n_2 = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x+1, y, HIDDEN]) else (-self.c*(2*self.pi[im, x+1, y] - 1)))
            right_n = ((self.c * (2 * self.pi[im, x + 1, y] - 1)))
            right_n_2 = ((-self.c * (2 * self.pi[im, x + 1, y] - 1)))

        up_n, up_n_2 = 0, 0
        if y - 1 >= 0:
            # up_n = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x, y-1, HIDDEN]) else (self.c*(2*self.pi[im, x, y-1] - 1)))
            # up_n_2 = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x, y-1, HIDDEN]) else (-self.c*(2*self.pi[im, x, y-1] - 1)))
            up_n = (self.c * (2 * self.pi[im, x, y - 1] - 1))
            up_n_2 = (-self.c * (2 * self.pi[im, x, y - 1] - 1))

        down_n, down_n_2 = 0, 0
        if y + 1 < self.side_length:
            # down_n = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x, y+1, HIDDEN]) else (self.c*(2*self.pi[im, x, y+1] - 1)))
            # down_n_2 = (0 if (self.mrf[im, x, y, HIDDEN] == self.mrf[im, x, y+1, HIDDEN]) else (-self.c*(2*self.pi[im, x, y+1] - 1)))
            down_n = ((self.c * (2 * self.pi[im, x, y + 1] - 1)))
            down_n_2 = ((-self.c * (2 * self.pi[im, x, y + 1] - 1)))

        first_term = left_n + right_n + up_n + down_n + visible_n
        second_term = left_n_2 + right_n_2 + up_n_2 + down_n_2 + visible_n_2
        return (np.exp(first_term), np.exp(second_term))

b_m = boltzmann_machine()


#%%  %%% plt scatter point over multile thresholds 
class ROC_curve(): 
    def __init__(self,original,denoised): 
        self.predicted = denoised 
        self.true = original 
        self.perf_measure() 
    def perf_measure(self):
        y_actual=self.true
        y_hat=self.predicted 
        print(y_actual[0:10] == y_hat[0:10])
        TP,FP,TN,FN = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1) 
        for i in range(0, len(y_hat)):   
            #print(y_actual[i],y_hat[i])
            if y_actual[i]==y_hat[i] and y_hat[i] ==1.0:
               TP += 1
            if y_hat[i]==1.0 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i] and y_hat[i] ==-1:
               TN += 1
            if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
               FN += 1  
        print(TP,FN,FP,TN)
        self.TPR= (TP/(TP+FN))
        self.FPR =(FP/((FP+TN)))  
        self.accuracy = (TP+TN)/((FP+TN)+(TP+FN))
        return self.TPR,self.FPR,self.accuracy  
def plot_ROC(FPR,TPR): 
    plt.figure()
    plt.plot(FPR,TPR)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
#%% first test was using differnt values of c while c2 = 0.2 whihc did not work 
    # except when c = 0,0.2 
 # second  test is to make c1,c2 equal each other 
    # {-1,0,0.2,1,2} which only worked for 0,0.2
# final try is to only change c2 while c1 is = 0.2
    # c2 = {-1,0,0.2,1,2} 

FPR_C2,TPR_C2,acc_C2= [],[],[] 
#%%
plot=  ROC_curve(original =np.reshape(b_m.orig_train_images,(392000)),
                           denoised = np.reshape(b_m.denoised_images,(392000)))  
FPR_C2.append((plot.FPR))
TPR_C2.append(plot.TPR) 
acc_C2.append(plot.accuracy)
#%$% 
#%% 
x= np.reshape(b_m.orig_train_images,(392000)) 

y=np.reshape(b_m.denoised_images,(392000))  
#%% 
print(b_m.accuracies)
#%% 
print(x[0:10]) 
print(y[0:10])
#%% 
plot_ROC(FPR_C2,TPR_C2) 
#%%  c2 = {-1,0,0.2,1,2}   
####### final results for c2 thresholding ##########
FPR =[0.9809,1,0.00393,0.019,0.017] 
TPR=[0.0194,1.0,0.989,0.985,0.989] 
 
 
#%% 
print(TPR_C2,FPR_2) 
#%% 
noisy = b_m.noisy_train_images  
orig = b_m.orig_train_images   
denoisy = b_m.denoised_images
plt.figure(1)
plt.subplot(3, 1, 1)
plt.imshow(orig[0, :, :])
plt.subplot(3, 1, 2)
plt.imshow(noisy[0, :, :])
plt.subplot(3, 1, 3)
plt.imshow(denoisy[0, :, :])
plt.show(1)