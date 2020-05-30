import pandas as pd
a = pd.read_csv('hPara.csv')
try:
    if a.iloc[-1,3]==None:
        exit()
    if a.iloc[-1,3] > 99.00:
        print("Successfull")
        exit()
except:
    print("Run AutoML.py first")
    exit()
max_epochs = 3
max_CRP = 3
max_Dense = 3

if a.iloc[-1,2] < max_epochs:
    newE = a.iloc[-1,2]+1
    newD = a.iloc[-1,1]
    newC = a.iloc[-1,0]
    print("Epoch Increased")
elif a.iloc[-1,1] < max_Dense:
    newD = a.iloc[-1,1]+1
    newE = 1
    newC = a.iloc[-1,0]
    print("Dense Added")
elif a.iloc[-1,0] < max_CRP:
    newC = a.iloc[-1,0]+1
    newD = 1
    newE = 1
    print("CRP Added")
else:
    print("TestedAll")
    
r = {'no_CRP_layers':newC, 'no_Dense_layers':newD , 'epochs':newE, 'acc':None}
a = a.append(r,ignore_index=True,sort=False)
a.to_csv('hPara.csv',index=False)

