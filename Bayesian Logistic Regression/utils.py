import numpy as np
import pystan

SEED = [171690, 349459, 697364]

# STAN NUTS
def count_divergences(fit):
    divergent_per_chain = np.column_stack([y['divergent__'] for y in fit.get_sampler_params(inc_warmup=False)]).sum(axis=0)
    return divergent_per_chain.sum()

def compute_metrics(w,b,X_test,y_test):
    """
    Calculates accuracy, precision, recall, and F1 score.
    :param w: coefficients of the linear combination for logistic regression
    :param b: bias coefficient of the linear combination for logistic regression
    :param X_test: descriptive variables values 
    :param y_test: true class assignments
    :return: accuracy, precision, recall, F1 score
    """
    y_pred = np.dot(X_test,w) + b
    y_pred = y_pred.reshape(y_test.shape)
    
    # equialent to computing logistic(np.dot(X_test,w) + b) >= 0.5
    y_pred[y_pred>=0] = 1
    y_pred[y_pred<0]  = 0

    total = y_test.shape
    TP = np.sum(y_pred * y_test)            # True positive rate
    FP = np.sum(y_pred - y_pred * y_test)   # False positive rate
    TN = np.sum((1-y_pred) * (1-y_test))    # True negative rate
    FN = total - TP - FP - TN               # False negative rate 
    
    accuracy = ((TP + TN) / total)[0]           
    precision = (TP / (TP + FP))
    recall = (TP / (TP + FN))[0]
    F1 = (2 / (1/recall + 1/precision))
    return accuracy, precision, recall, F1