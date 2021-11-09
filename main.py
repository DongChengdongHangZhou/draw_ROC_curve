from numpy.core.function_base import logspace
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

gan_name = 'stargan'
score_real = pd.read_csv('./'+gan_name+'/'+gan_name+'_real.csv',header=None)
score_fake = pd.read_csv('./'+gan_name+'/'+gan_name+'_fake.csv',header=None)
score_method1 = pd.read_csv('./'+gan_name+'/'+gan_name+'_method1.csv',header=None)
score_method2 = pd.read_csv('./'+gan_name+'/'+gan_name+'_method2.csv',header=None)
score_method3 = pd.read_csv('./'+gan_name+'/'+gan_name+'_method3.csv',header=None)
score_real = np.array(score_real)[0].tolist()
score_fake = np.array(score_fake)[0].tolist()
score_method1 = np.array(score_method1)[0].tolist()
score_method2 = np.array(score_method2)[0].tolist()
score_method3 = np.array(score_method3)[0].tolist()
label_real = (np.ones((1999,))*1).tolist()
label_fake = (np.ones((1999,))*2).tolist()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(2.718, adjustable='box')
plt.rcParams.update({'font.size':20})

def draw_roc(logscale):
    y_label = label_fake + label_real
    y_pre1 = score_fake + score_real
    y_pre2 = score_method1 + score_real
    y_pre3 = score_method2 + score_real
    y_pre4 = score_method3 + score_real
    fpr1, tpr1, thersholds1 = roc_curve(y_label, y_pre1, pos_label=2)
    fpr2, tpr2, thersholds2 = roc_curve(y_label, y_pre2, pos_label=2)
    fpr3, tpr3, thersholds3 = roc_curve(y_label, y_pre3, pos_label=2)
    fpr4, tpr4, thersholds4 = roc_curve(y_label, y_pre4, pos_label=2)
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'r--', label='unprocessed ROC (area = {0:.2f})'.format(roc_auc1), lw=2)
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, 'b--', label='method1 ROC (area = {0:.2f})'.format(roc_auc2), lw=2)
    roc_auc3 = auc(fpr3, tpr3)
    plt.plot(fpr3, tpr3, 'g--', label='method2 ROC (area = {0:.2f})'.format(roc_auc3), lw=2)
    roc_auc4 = auc(fpr4, tpr4)
    plt.plot(fpr4, tpr4, 'k--', label='method3 ROC (area = {0:.2f})'.format(roc_auc4), lw=2)
    plt.xlim([0.001, 1.05])  
    plt.ylim([0.001, 1.05])

    if logscale == True:
        plt.xscale('log')
        # plt.yscale('log')
    
    plt.grid(True,which="both", linestyle='--')
    plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman', 'size': 21})
    plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman', 'size': 21})  
    plt.yticks(fontproperties = 'Times New Roman', size = 17)
    plt.xticks(fontproperties = 'Times New Roman', size = 17)
    plt.title('ROC Curve',fontdict={'family':'Times New Roman', 'size': 21})
    plt.legend(title=gan_name,loc="lower right",prop={'family' : 'Times New Roman', 'size'   : 19})
    plt.show()

if __name__ == '__main__':
    logscale = True
    draw_roc(logscale)
