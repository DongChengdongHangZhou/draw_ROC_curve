from numpy.core.function_base import logspace
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':25})
def draw_roc(logscale):
    y_label = ([1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2 ,2, 2]) 
    y_pre1 = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6,0.32, 0.52, 0.92, 0.82, 0.42, 0.62,0.34, 0.54, 0.94, 0.84, 0.44, 0.64,0.36, 0.56, 0.96, 0.86, 0.46, 0.66,0.1, 0.27, 0.93, 0.83, 0.43, 0.63,0.15, 0.55, 0.95])
    y_pre2 = ([0.35, 0.5, 0.9, 0.8, 0.4, 0.6,0.32, 0.52, 0.92, 0.82, 0.48, 0.62,0.34, 0.54, 0.94, 0.84, 0.44, 0.64,0.36, 0.56, 0.96, 0.86, 0.46, 0.66,0.1, 0.2, 0.93, 0.83, 0.43, 0.63,0.15, 0.55, 0.4])
    y_pre3 = ([0.1, 0.5, 0.9, 0.8, 0.4, 0.6,0.32, 0.52, 0.92, 0.82, 0.48, 0.62,0.34, 0.54, 0.94, 0.84, 0.44, 0.64,0.36, 0.56, 0.01, 0.86, 0.46, 0.66,0.1, 0.2, 0.93, 0.83, 0.43, 0.63,0.15, 0.55, 0.4])
    y_pre4 = ([0.9, 0.5, 0.9, 0.8, 0.1, 0.6,0.32, 0.52, 0.92, 0.82, 0.48, 0.62,0.34, 0.54, 0.94, 0.84, 0.14, 0.64,0.01, 0.56, 0.96, 0.16, 0.46, 0.66,0.1, 0.2, 0.93, 0.83, 0.43, 0.63,0.15, 0.15, 0.4])
    fpr1, tpr1, thersholds1 = roc_curve(y_label, y_pre1, pos_label=2)
    fpr2, tpr2, thersholds2 = roc_curve(y_label, y_pre2, pos_label=2)
    fpr3, tpr3, thersholds3 = roc_curve(y_label, y_pre3, pos_label=2)
    fpr4, tpr4, thersholds4 = roc_curve(y_label, y_pre4, pos_label=2)
    for i, value in enumerate(thersholds1):
        print("%f %f %f" % (fpr1[i], tpr1[i], value))
    
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'r--', label='unprocessed ROC (area = {0:.2f})'.format(roc_auc1), lw=2)
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, 'b--', label='method1 ROC (area = {0:.2f})'.format(roc_auc2), lw=2)
    roc_auc3 = auc(fpr3, tpr3)
    plt.plot(fpr3, tpr3, 'g--', label='method2 ROC (area = {0:.2f})'.format(roc_auc3), lw=2)
    roc_auc4 = auc(fpr4, tpr4)
    plt.plot(fpr4, tpr4, 'k--', label='method3 ROC (area = {0:.2f})'.format(roc_auc4), lw=2)
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])

    if logscale == True:
        plt.xscale('log')
        # plt.yscale('log')
    
    plt.grid(True,which="both", linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  
    plt.title('ROC Curve')
    plt.legend(title='GauGAN',loc="lower right")
    plt.show()

if __name__ == '__main__':
    logscale = False
    draw_roc(logscale)
