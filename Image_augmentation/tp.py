import pandas as pd
train_df=pd.read_csv('./mitbih_train.csv',header=None)
train_df[187]=train_df[187].astype(int)

count1=0
count2=0
count3=0
count4=0
count5=0

for rows in train_df[187]:
    if rows ==0:
        count1=count1  +1
    if rows ==1:
        count2=count2  +1
    if rows ==2:
        count3=count3  +1
    if rows ==3:
        count4=count4  +1
    if rows ==4:
        count5 = count5+1
print(count1,count2,count3,count4,count5)