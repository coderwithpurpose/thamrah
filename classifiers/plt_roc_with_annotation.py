
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import mean_squared_error
def plot_ROC(FPR,TPR): 
    plt.figure()
    plt.scatter(FPR,TPR) 
    labels = [-1,0,0.2,1,2]

    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        FPR, TPR, marker='o', 
        cmap=plt.get_cmap('Spectral'))

    for label, x, y in zip(labels, FPR, TPR):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlim([0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
#         plt.legend(loc="lower right")
    plt.show()
    #%% first test was using differnt values of c while c2 = 0.2 whihc did not work 
        # except when c = 0,0.2 


    #%% 
    #%%  c2 = {-1,0,0.2,1,2}   
    ####### final results for c2 thresholding ##########
    # FPR =[0.9809,1,0.00393,0.019,0.017] 
    # TPR=[0.0194,1.0,0.989,0.985,0.989]
# TPR=[0.524,0.98,0.989,0.984,0.983] 
# FPR =[0.530,0.023,0.005,0.005,0.005] 
TPR=[0.4,0.45,0.989,0.984,0.983] 
FPR =[0.850,0.45,0.005,0.005,0.005] 
plot_ROC(FPR,TPR) 

