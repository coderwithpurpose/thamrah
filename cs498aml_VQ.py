
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

class acc_classifier:

    def __init__(self, data_dir):
        self.get_activity_data(data_dir)
        self.get_training_and_testing_data()
        self.cluster_data()
        self.train_data_hists = self.quantize_data(self.train_data)
        ## added
        self.test_data_hists = self.quantize_data(self.test_data)
        pass

    def get_activity_data(self, data_dir, segment_size=32):

        # First, we iterate through all the needed directories
        directory_list_w_MODEL, directory_list = os.listdir(data_dir), []
        directory_list = [x for x in directory_list_w_MODEL if 'MODEL' not in x and '.txt' not in x and '.m' not in x and '.DS_Store' not in x]
        sample_data, data = {}, {}
        file_counter = 0
        for sample_dir in directory_list:
            sample_data[sample_dir] = []
            for file in os.listdir(data_dir + sample_dir + '/'):
                file_counter += 1
                # if sample_data[sample_dir] == []:
                #     sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
                # else:
                #     # sample_data[sample_dir] = np.concatenate((sample_data[sample_dir], np.genfromtxt(data_dir + sample_dir + '/' + file)), axis=0)
                #     sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
                sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
            data[sample_dir] = []
            for i in np.arange(len(sample_data[sample_dir])):
                obs_data, new_obs_data = sample_data[sample_dir][i], []
                for j in np.arange(int(obs_data.shape[0]/segment_size)):
                    segment = obs_data[j*segment_size:(j+1)*segment_size, :].flatten()[:, np.newaxis]
                    if new_obs_data == []:
                        new_obs_data = segment
                    else:
                        new_obs_data = np.concatenate((new_obs_data, segment), axis=1)
                data[sample_dir] += [new_obs_data.T]
        # print(file_counter)
        self.data = data
        self.sample_data = sample_data

    def get_training_and_testing_data(self):
        self.train_data, self.test_data, self.clustering_data = {}, {}, []
        for key in self.data.keys():
            train_split = int(0.90 * len(self.data[key]))
            arr = self.data[key][:train_split]
            # if self.clustering_data == []:
            #     self.clustering_data = arr
            # else:
            #     self.clustering_data = np.concatenate((self.clustering_data, arr), axis=0)
            self.clustering_data += arr
            self.train_data[key] = arr
            self.test_data[key] = self.data[key][train_split:]
        clustered_train_data = []
        for i in range(len(self.clustering_data)):
            signal = self.clustering_data[i]
            if clustered_train_data == []:
                clustered_train_data = signal
            else:
                clustered_train_data = np.concatenate((clustered_train_data, signal), axis=0)
        self.clustering_data = clustered_train_data

    def cluster_data(self, num_clusters_1st_lvl=16, num_clusters_2nd_lvl=3):

        # First, we create the initial top level clusters
        KMeans_model_1st_lvl = KMeans(n_clusters=num_clusters_1st_lvl,
                                init='k-means++',
                                random_state=0
                                )
        # We then fit the data to the the first level Kmeans model
        KMeans_model_1st_lvl.fit(self.clustering_data)
        predictions = KMeans_model_1st_lvl.predict(self.clustering_data)
        first_lvl_cluster_centers = {}
        for cluster_num in np.arange(KMeans_model_1st_lvl.n_clusters):
            curr_cluster_data = self.clustering_data[cluster_num == predictions, :]
            KMeans_model_2nd_lvl =  KMeans(n_clusters=num_clusters_2nd_lvl,
                   init='k-means++',
                   random_state=0
                   )
            KMeans_model_2nd_lvl.fit(curr_cluster_data)
            second_lvl_cluster_centers = {}
            for second_lvl_cluster_num in np.arange(KMeans_model_2nd_lvl.n_clusters):
                cluster_center = KMeans_model_2nd_lvl.cluster_centers_[second_lvl_cluster_num]
                second_lvl_cluster_centers[second_lvl_cluster_num] = (second_lvl_cluster_num, cluster_center)
            first_lvl_curr_center = KMeans_model_1st_lvl.cluster_centers_[cluster_num]
            first_lvl_cluster_centers[cluster_num] = (cluster_num, first_lvl_curr_center, second_lvl_cluster_centers)
        self.cluster_dictionary = first_lvl_cluster_centers

    def quantize_data(self, data, num_clusters_1st_lvl=16, num_clusters_2nd_lvl=3):

        quantized_data = {}
        histograms = []
        for act in data.keys():
            # print(act)
            for sample in np.arange(len(data[act])):
                quantized_data_sample = np.zeros(num_clusters_1st_lvl * num_clusters_2nd_lvl)
                for i in np.arange(data[act][sample].shape[0]):
                    hist_idx = self.return_closest_cluster_index(data[act][sample][i])
                    quantized_data_sample[hist_idx] += 1
                histograms += [quantized_data_sample]
            quantized_data[act] = histograms
        return quantized_data

    def return_closest_cluster_index(self, data_sample):

        min_idx, min_dist = float("inf"), float("inf")
        for cluster_num, center in self.cluster_dictionary.items():
            dist = np.linalg.norm(data_sample - center[1])
            if dist < min_dist:
                min_idx, min_dist = cluster_num, dist
        min_idx_2, min_dist_2 = float("inf"), float("inf")
        for cluster_num, center in self.cluster_dictionary[min_idx][2].items():
            dist = np.linalg.norm(data_sample - center[1])
            if dist < min_dist_2:
                min_idx_2, min_dist_2 = cluster_num, min_dist_2
        return min_idx * min_idx_2





data_dir = 'HMP_Dataset/'
obj = acc_classifier(data_dir)

#
test = obj.train_data_hists
dict2 = test.copy()
test2 = obj.test_data_hists
dict3 = test2.copy()



hists_train = {}
labels_train = {}
hists_test = {}
labels_test = {}
flag = 0
i = 0

# thsoe 2 dicitonaries will be used to map the string labels to numebrs 
labels_dict = {'class': 'cs498'}
# true_labels = {'class' : 'cs498'}

## preparing the trining data
for act in dict2.keys():
    temp_dic = {act:i}
    labels_dict.update(temp_dic)
    
    # hold labels of each histogram
    # first get all histograms of the same label
    a = dict2[act] 
    # second make an array with same length having the same label
    w,l = np.shape(a)
    ones = np.ones(w)
    new_mtx = ones * i
    i += 1
    
    # append the histogram and label matrices by the new actions infos 
    if flag !=0:
        hists_train = np.concatenate((hists_train, a), axis=0)
        labels_train = np.concatenate((labels_train, new_mtx), axis=0)

    else:
        labels_train = new_mtx
        hists_train = a
        flag = 1

flag = 0

# prepairing the testing data in the same fashion as the trainng set 
# but now we use the same labels dictionary for mapping 
for act in dict3.keys():
    i = labels_dict[act]
    a = dict3[act]
    w,l = np.shape(a)
    ones = np.ones(w)
    new_mtx = ones * i
    if flag !=0:
        hists_test = np.concatenate((hists_test, a), axis=0)
        labels_test = np.concatenate((labels_test, new_mtx), axis=0)
    else:
        labels_test = new_mtx
        hists_test = a
        flag = 1

# print("training data : ", np.shape(hists_train), np.shape(labels_train), labels_train, hists_train)
# print("testing data : ", np.shape(hists_test), np.shape(labels_test), labels_test)


# # building the random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# now bulding a classifier assuming we have labels correspond to each histogram
trained_model = RandomForestClassifier(max_features = None)
trained_model.fit(hists_train, labels_train)
# print "Trained model :: ", trained_model

#predict over test histograms
predictions = trained_model.predict(hists_test)


# # check prediction resulrs + confusion matrix
print((labels_test[:20]))
print((predictions[:20]))
print "Test Accuracy  :: ", accuracy_score(labels_test, predictions)
# print " Confusion matrix ", confusion_matrix(labels_test, predictions)


import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=test2.keys(),
                      title='Confusion matrix, without normalization')


## to DO 
# remap labels and fix the classifier issue  

# what i have done 
'''
first i put all histograms of the same activity into variable called hists_train where for 
each activity there is hists of (748,48) so the variable would be (748*14,48), and labels 
with same sequence of (14*748) where each 748 block is of the same label. 
i mapped the labels from text to numbers to be able to classify them numerically  
and for the test data i added the line in the init stage of the main class, after that i just followed the same 
procedure. 
''' 

# what could went wrong is probably the way i prepared the trning and testing data. 
# if not then it's an issue with i inilized the classifier 
# i have also noted that even if i pass to the classifier exactly the same matrix as the one used for training 
# i get exactly the same error rate 

