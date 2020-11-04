import Parameter as para
import pandas as pd
import numpy as np

class Regression:
    def __init__(self, startAt):

        self.E_ele = []
        self.delta = []
        self.M_ele = []
        self.startAt = startAt
    def read_data(self, train_filename, target_filename):

        data = pd.read_csv(train_filename)
        target = pd.read_csv(target_filename)
        self.E_ele = np.asarray(data['E_ele'])
        self.M_ele = np.asarray(data['M_ele'])
        self.delta = np.asarray(target['delta'])

    def update(self):

        idx = self.startAt + para.X
        train_data = np.array([(e, m) for e, m in zip(self.E_ele[self.startAt: idx], self.M_ele[self.startAt: idx])])
        target_data = self.delta[self.startAt: idx]
        #target_data = target_data/sum(target_data)
        A = np.dot(train_data.T, train_data)
        b = np.dot(train_data.T, target_data)
        W = np.dot(np.linalg.pinv(A), b)
        self.startAt = idx

        return W






