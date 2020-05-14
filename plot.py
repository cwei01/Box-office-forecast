import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':
    data = pd.read_csv('data/feature_data.csv',encoding='utf-8')
    data.fillna('0', inplace=True)
    # -----------budget----------------------------------
    # plt.plot(data['budget'],data['revenue'],'o')
    # plt.xlabel('budget')
    # plt.ylabel('revenue')
    # plt.title('budget-revenue')
    # plt.savefig('budget-revenue.png')
    #plt.show()
    #-----------genres----------------------------------
    # plt.plot(data['original_language'], data['revenue'], 'o')
    # plt.xlabel('original_language')
    # plt.ylabel('revenue')
    # plt.title('original_language-revenue')
    # plt.savefig('original_language-revenue.png')
    # plt.show()
    #-----------popularity----------------------------------
    # plt.plot(data['popularity'], data['revenue'], 'ro')
    # plt.xlabel('popularity')
    # plt.ylabel('revenue')
    # plt.title('popularity-revenue')
    # plt.savefig('popularity-revenue.png')
    # plt.show()

    #----------train_data---------------
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

    x=[0.2,0.4,0.6,0.7]
    y1=[1799236,2017159,2378978,2855316]
    y2=[2610350,2821755,3081218,3505899]
    y3=[780804,870224,2452395,1950035]
    plt.plot(x,y1,'-',label='LR')
    plt.plot(x,y2,'-',label='DART')
    plt.plot(x,y3,'-',label='NN')
    plt.legend()
    plt.xlabel('train_data_rate')
    plt.ylabel('MSE')
    plt.savefig('MSE.png')
    plt.show()