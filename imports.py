import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import sklearn
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
import DT as dt
import SVM as svm
import RandomForest as rf
import NavieBayes as nb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from LearningCurve import plot_learning_curve
from LearningCurve import plot_validation_curve
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')