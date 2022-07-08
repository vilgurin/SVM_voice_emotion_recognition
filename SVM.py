import numpy as np
import random
import os
import soundfile
import glob
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SupportVectorMachine:
    
    def __init__(self, C, kernel):
        self.C = 1                      
        self.kernel = kernel      
        self.alphas = None
        self.errors = None
        self.epsilon = 0.01
        self.b = 0
           
    def fit(self, trainX, trainY):
        self.alphas = np.zeros(np.shape(trainX)[0])
        self.x = np.array(trainX)
        self.y = np.array(trainY)
        initial_error = np.array(self.error_init(self.alphas, self.y, self.kernel,self.x, self.x, self.b) - self.y)
        self.errors = np.array(initial_error)
        self.train()
        return self.alphas

    def predict_one(self,X):
        result = self.kernel(X,self.support_vectors)
        result = result * self.supportAlphaY
        return np.sum(result)

    def predict(self, X):
        take_vectors = self.alphas > 1e-6
        self.support_vectors = self.x[take_vectors]
        self.supportAlphaY = self.y[take_vectors] * self.alphas[take_vectors]
        distance = []
        for i in range(len(X)):
            distance.append(self.predict_one(X[i]))
        return distance

    def obj_f(self,alphas,y,X):
        s_alphas = np.sum(alphas)
        s = 0
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                s += y[i]*y[j]*alphas[i]*alphas[j]*self.kernel(X[i],X[j])
        return s_alphas - 0.5 * s

    def compute_w(self):
      s = 0
      for i in range(len(self.alphas)):
        s += self.x[i]*self.y[i]*self.alphas[i]
      return s
    def error_init(self,alphas, target, kernel, X_train, x_test, b):
    
        result = np.matmul((alphas * target),kernel(X_train, x_test)) + b
        return result

    def classifier(self,alpha, X, X_i, y_i, b):
      return alpha * y_i * self.kernel(X_i, X) + b

    def calculate_error_cache(self,alpha1,alpha2,b,range,i1,i2):
        self.errors[range] = self.errors[range] + self.classifier(alpha1,self.x[range],self.x[i1],self.y[i1],self.b) + self.classifier(alpha2,self.x[range],self.x[i2],self.y[i2],self.b) - self.b -b


    def compute_LH(self,alpha1,alpha2,y1,y2):
        if y1 != y2:
            L = max(0,alpha2-alpha1)
            H = min(self.C,self.C+alpha2-alpha1)
        else:
            L = max(0,alpha1+alpha2-self.C)
            H = min(self.C,alpha1+alpha2)
        return L,H

    def positive_eta(self,alphas,i,L,H,epsilon,alpha2):
        new_alphas = alphas.copy()
        new_alphas[i] = L
        L_objective = self.obj_f(new_alphas,self.y,self.x)
        new_alphas[i] = H
        H_objective = self.obj_f(new_alphas,self.y,self.x)

        if L_objective < H_objective-epsilon:
            a2 = L
        elif  L_objective > H_objective+epsilon:
            a2 = H
        else:
            a2 = alpha2
        return a2

    def choose_b(self,a1,a2,b1,b2):
        if a1 > 0 and a1 < self.C:
            return b1
        elif a2 > 0 and a2 < self.C:
            return b2
        else: 
            return (b1+b2)/2

    def check_bounds(self,alpha):
        if alpha > 0  and alpha < self.C:
            return 0

    def take_step(self,i1,i2):
        if i1 == i2:
            return 0

        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]

        E1 = self.errors[i1]
        E2 = self.errors[i2]

        L_an_H = self.compute_LH(alpha1,alpha2,y1,y2)
        L = L_an_H[0]
        H = L_an_H[1]
        

        if L == H:
            return 0
        
        # kernel and eta
        k11 = self.kernel(self.x[i1],self.x[i1])
        k12 = self.kernel(self.x[i1],self.x[i2])
        k22 = self.kernel(self.x[i2],self.x[i2])
        eta =  2*k12 - k11 - k22 
        if eta < 0:
            a2 = alpha2 - y2 *(E1-E2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            a2 = self.positive_eta(self.alphas,i2,L,H,self.epsilon,alpha2)

        if np.abs(a2-alpha2) < self.epsilon*(a2+alpha2+self.epsilon):
            return 0
        a1 = alpha1 + y1*y2*(alpha2-a2)

        b1 = E1+self.b +y1*(a1-alpha1)*k11+y2*(a2-alpha2)*k12
        b2 = E2+self.b +y1*(a1-alpha1)*k12+y2*(a2-alpha2)*k22
        b = self.choose_b(alpha1,alpha2,b1,b2)

        self.alphas[i1] = a1
        self.alphas[i2] = a2
        # update error cache
        if a1 > 0 and a1 < self.C:
            self.errors[i1] = 0.0
        if a2 >0 and a2 < self.C:
            self.errors[i2] = 0.0

        without_i = [x for x in range(len(self.alphas)) if x != i1 and x!= i2]
        self.calculate_error_cache(a1-alpha1,a2-alpha2,b,without_i,i1,i2)

        self.b = b
        return 1


    def examine_example(self,i2):
      tol = 0.01
      y2 = self.y[i2]
      alpha2 = self.alphas[i2]
      E2 = self.errors[i2]
      r2 = E2 * y2
      if (r2 < -tol and alpha2 <self.C) or (r2 > tol and alpha2 > 0):
        if self.errors[i2] > 0:
            index = np.where(self.errors >= min(self.errors))[0][0]
        elif self.errors[i2] < 0:
            index = np.where(self.errors >= max(self.errors))[0][0]
        if self.take_step(index,i2):
            return 1

        random_point = random.randint(0,len(self.alphas))
          # non-zero alphas
        for i in range(random_point, len(self.alphas)):
            if self.alphas[i] != 0 and self.alphas[i] != self.C:
                if self.take_step(i,i2):
                    return 1

        random_point = random.randint(0,len(self.alphas))
        # all alphas
        for i in range(random_point, len(self.alphas)):
           if self.take_step(i,i2):
               return 1
      return 0

    def train(self):
        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(len(self.alphas)):
                    changed_example = self.examine_example(i)
                    num_changed += changed_example
            else:
                for i in range(len(self.alphas)):
                    if self.alphas[i] != 0 and self.alphas[i] != self.C:
                        changed_example = self.examine_example(i)
                        num_changed += changed_example
            if examine_all:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1   

def kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return np.exp(-np.power(np.linalg.norm(x-y),2)/2*sigma**2)
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
      result = []
      temp = []
      for i in range(len(x)):
        for j in range(len(y)):
          diff = x[i]-y[j]
          temp_result = np.exp(-np.power(np.linalg.norm(diff),2)/2*sigma**2)
          temp.append(temp_result)
        result.append(temp)
        temp = []
      return np.array(result)
    else:
        return np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))