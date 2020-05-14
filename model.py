import  pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")
import time

def Neural_etwork(X_train,X_test, y_train, y_test):
    score_list=[]
    # for i in range(1,10):
    #     ANNmodel_test=MLPClassifier(activation='tanh', hidden_layer_sizes=i,batch_size=200)
    #     ANNmodel_test.fit(X_train, y_train)
    #     score = ANNmodel_test.score(X_train, y_train)
    #     score_list.append(score)
    # t=np.argmax(score_list)+1
    ANNmodel = MLPClassifier(activation='tanh', hidden_layer_sizes=1, verbose=False, batch_size=400)
    starttime = time.time()
    ANNmodel.fit(X_train, y_train)
    y_pred=ANNmodel.predict(X_test)
    endtime = time.time()
    dtime=endtime-starttime
    sum = 0
    for i in range(len(y_pred)):
        sum += (y_pred[i] - y_test[i]) ** 2
    sum_error = (np.sqrt(sum)) / len(y_pred)
    return sum_error,round(dtime,4)


def LR(X_train,X_test, y_train, y_test):
    linreg = LinearRegression()
    starttime = time.time()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    endtime = time.time()
    dtime=endtime-starttime
    sum = 0
    for i in range(len(y_pred)):
        sum += (y_pred[i] - y_test[i]) ** 2
    sum_error = (np.sqrt(sum))/len(y_pred)
    return int(sum_error),round(dtime,4)
def DART(X_train,X_test, y_train, y_test):
    dtr=DecisionTreeRegressor()
    starttime=time.time()
    dtr.fit(X_train,y_train)
    y_pred=dtr.predict(X_test)
    endtime=time.time()
    dtime=endtime-starttime
    sum = 0
    for i in range(len(y_pred)):
        sum += (y_pred[i] - y_test[i]) ** 2
    sum_error = (np.sqrt(sum)) / len(y_pred)
    return int(sum_error),round(dtime,4)


if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv', encoding='utf-8')
    test_data = pd.read_csv('data/test.csv', encoding='utf-8')
    train_X=train_data.iloc[:,1:-1]
    train_Y=train_data['revenue']
    test_X = test_data.iloc[:, 1:-1]
    test_Y = test_data['revenue']
    print(LR(train_X,test_X,train_Y,test_Y))
    print(DART(train_X,test_X,train_Y,test_Y))
    print(Neural_etwork(train_X,test_X,train_Y,test_Y)) #1954138

    #(2855316, 0.0239)   7
    #(3505899, 0.0249)
    #(1950035, 41.4388)

    # (2017159, 0.006)  4
    # (2821755, 0.012)
    # (870224.9566744977, 5.0141)

    # (1799236, 0.003)  2
    # (2610350, 0.007)
    # (780804.4535338144, 0.467)

    # (2378978, 0.004)  6
    # (3081218, 0.017)
    # (2452395.1157750166, 33.1885)
