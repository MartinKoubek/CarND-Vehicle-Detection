'''
Created on 23. 8. 2017

@author: ppr00076
'''
from sklearn.svm import LinearSVC

class Classifier(object):
    def __init__(self, type = 'SVC'):
        pass

    def run(self, X_train, y_train):
        self.svc = LinearSVC()
        self.svc.fit(X_train, y_train)
        
        
    def getAccuracy(self, X_test, y_test):
        accuracy = round(self.svc.score(X_test, y_test),4)
        return accuracy
    
    def predict(self, test_features):
        return self.svc.predict(test_features)
    
if __name__ == '__main__':
    pass