from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# import plotting
import numpy as np
from tqdm.notebook import tqdm

def plot_roc(ref_model, hls_model, X_npy, y_npy, output_dir=None, data_split_factor=1):
    '''
    receives a keras model and an hls_model and plots 
    the roc_curve against the X_npy and y_npy. A plot
     is also created at output_dir if it is provided
    '''
    #load processed test data
    X = np.load(X_npy, allow_pickle=True)
    y = np.load(y_npy, allow_pickle=True)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

    # use a quarter of the test_set to save time
    for i in range(len(X)):
        divider = int(len(X[i])/data_split_factor)
        assert len(X) == len(y)
        X[i], y[i] = shuffle(X[i], y[i])
        X[i], y[i] = X[i][0:divider],  y[i][0:divider]
    print("using one quarter of provided dataset for roc plot")

    fig, ax = plt.subplots(figsize=(9, 9))
    #perform inference
    # with tqdm(total=total, position=0, leave=True) as pbar:
    #     for i in tqdm((foo_, range_ ), position=0, leave=True):
    #     pbar.update()
    for index, X_data in enumerate(tqdm(X)): # position=0, leave=True))
        ref_pred = [0. for ind in X_data]
        QDenseBN_pred = [0. for ind in X_data]
        for file_idx, X_test in enumerate(tqdm(X_data)):
            ref_predictions = ref_model.predict(X_test)
            ref_errors = np.mean(np.square(X_test-ref_predictions), axis=1)
            ref_pred[file_idx] = np.mean(ref_errors)
            
            QDenseBN_predictions = hls_model.predict(X_test)
            QDenseBN_errors = np.mean(np.square(X_test-QDenseBN_predictions), axis=1)
            QDenseBN_pred[file_idx] = np.mean(QDenseBN_errors)
            # tqdm._instances.clear()
            
        #generate auc and roc metrics
        y_test = y[index]
        k_fpr, k_tpr, k_threshold = metrics.roc_curve(y_test, ref_pred)
        k_roc_auc = metrics.auc(k_fpr, k_tpr)
        h_fpr, h_tpr, h_threshold = metrics.roc_curve(y_test, QDenseBN_pred)
        h_roc_auc = metrics.auc(h_fpr, h_tpr)


        plt.title('Receiver Operating Characteristic')
        plt.plot(k_fpr, k_tpr, label = 'QDense AUC m_{} = {}'.format(index, round(k_roc_auc,2)), linewidth = 1.5, color=colors[index])
        plt.plot(h_fpr, h_tpr, label = 'QDenseBN AUC m_{} = {}'.format(index, round(h_roc_auc,2)), linewidth = 1, linestyle='--', color=colors[index])
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--', linewidth=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    plt.show()
    if output_dir != None:
        plt.savefig('{}/qdense_vs_qdensebatchnorm_roc_curve'.format(output_dir))
        print("QDense vs QDenseBN plot saved in {}".format(output_dir))