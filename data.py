import re
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")
train_data='data/train.csv'
test_data='data/test.csv'
def split_data(file0,N,file1,file2):
    lines = open(file0, 'r', encoding='utf-8').readlines()
    lines_for_b = int(len(lines) * N)+1
    open(file1, 'w', encoding='utf-8').write(''.join(lines[:lines_for_b]))
    open(file2, 'w', encoding='utf-8').write(''.join(lines[lines_for_b:]))
    column_name = []
    data = pd.read_csv(file0, encoding='utf-8')
    for i in range(len(data.columns.values)):
        column_name.append(data.columns.values[i])
    test_data= pd.read_csv(file2, header=None, names=column_name)
    test_data.to_csv(file2, index=False)

if __name__ == '__main__':
    data = pd.read_csv('data/imdb.csv', encoding='utf-8')
    data.info()
    #-----------budget----------------------------------
    data.loc[data['budget'] == 0, 'budget'] = sum(data['budget'])/len(data['budget'])
    # -----------genres----------------------------------
    data['genres'].fillna(0, inplace=True) #空值用0填充
    for i in range(len(data['genres'])):
        if data['genres'][i]!=0:
           cost=re.findall(r'[1-9]+\.?[0-9]*',str(data['genres'][i]))
           data['genres'][i]=cost[0]
    # -----------original_language----------------------------------
    probs = data['original_language'].value_counts(normalize=True)
    for i in range(len(probs)):
      data.loc[data['original_language'] ==probs.keys()[i], 'original_language'] = round(probs[i],3)
    # -----------production_companies----------------------------------
    data['production_companies'].fillna(0, inplace=True)  # 空值用0填充
    for i in range(len(data['production_companies'])):
        if data['production_companies'][i] != 0:
            cost = re.findall(r'[1-9]+\.?[0-9]*', str(data['production_companies'][i]))
            data['production_companies'][i] = cost[0]
    # -----------Keywords-------------------------------------------
    data['Keywords'].fillna(0, inplace=True)  # 空值用0填充
    for i in range(len(data['Keywords'])):
        if data['Keywords'][i] != 0:
            cost = re.findall(r'[1-9]+\.?[0-9]*', str(data['Keywords'][i]))
            data['Keywords'][i] = cost[0]
    feature_data={'budget':data['budget'],
                  'genres':data['genres'],
                  'original_language':data['original_language'],
                  'popularity':data['popularity'],
                  'production_companies':data['production_companies'],
                  'runtime':data['runtime'],
                  'Keywords':data['Keywords'],
                  'revenue':data['revenue']}
    feature_data = pd.DataFrame(feature_data)
    #print(feature_data.isnull().any())
    feature_data.fillna('0',inplace=True)
    feature_data.to_csv('data/feature_data.csv')

    split_data('data/feature_data.csv',0.65,train_data,test_data)









