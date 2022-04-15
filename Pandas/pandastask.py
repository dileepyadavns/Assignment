
import pandas as pd
import numpy as np

#Reading the csv 

df=pd.read_csv("purchase_data.csv")

df=df.sort_values(by='Age')

print(df)

print(df['Age'].max()) # prints max value in age column

print(df['Age'].min())# prints min value in age column


list1=np.arange(6,50,4).tolist() #array 

print(list1)
#for forming bins of age difference 4.
df['bins'] = pd.cut(x=df['Age'], bins=list1)
print(df)

counts=df.value_counts('bins')  #Purchase count for Each group
print(counts)

df['bins'].value_counts() #no of bins

#selected first bin and applied mean ,sum,count on each bin
fstbin = df['bins']==pd.Interval(left=6,right=10)

df[fstbin]

avg1st=df[fstbin]['Price'].mean()
print(avg1st)                   # mean for 1st group (6,10] 
sum1st=df[fstbin]['Price'].sum() #sum for 1st group (6,10] 
count1=df[fstbin]["Purchase ID"].count()  #Purchase count  for 1st group (6,10] 
print(count1)
print(sum1st)

#Selected second bin from Dataframe

bin2= df['bins']==pd.Interval(left=10,right=14)  

df[bin2]

mean2=df[bin2]['Price'].mean() 
print(mean2)                       # mean  for 2nd group (10,14] 
sum2=df[bin2]['Price'].sum()  #total price  for 2nd group (10,14] 
print(sum1st)
count2=df[bin2]["Purchase ID"].count() # purchase count for 2nd group (10,14] 
print(count2)

bin3= df['bins']==pd.Interval(left=14,right=18)
print(df[bin3]['Price'].mean())  # mean of price  3rd group (14,18] 
sum3=df[bin3]['Price'].sum()  # sum of  3rd group (14,18] 
print(sum1st)
count3=df[bin3]["Purchase ID"].count()  # purchase count of  3rd group (14,18] 
print(count3)

#selected 4th bin from data frame

bin4= df['bins']==pd.Interval(left=18,right=22)
                                                  
print(df[bin4]['Price'].mean()) # mean of price 4th group (18,22]
sum4=df[bin4]['Price'].sum() # total purchase price in 4th group (18,22]
print(sum4)
count4=df[bin4]["Purchase ID"].count() # purchase count  for 4th group (18,22]
print(count4)

#Selected fifth bin from Dataframe
bin5= df['bins']==pd.Interval(left=22,right=26)
                                                
print(df[bin5]['Price'].mean())  # mean of price for 5rd group (22,26] 
sum5=df[bin5]['Price'].sum()  # total purchase price   for 5rd group (22,26] 
print(sum5)

count5=df[bin5]["Purchase ID"].count()  # Purchase count  for 5rd group (22,26] 
print(count5)



print(df)

df['Purchase ID'].count() # purchase count for whole datset

df['Price'].mean() # average price for whole dataset

df['Price'].sum() # Total Purchase price for whole dataset

#Top five spenders

new=df.nlargest(5,["Price"]) #top five spenders

print(new[['SN',"Price"]])

print(new["Price"].mean() ) # average purchase value of top 5 spenders

print(new['Price'].sum())#total Purchase value of top 5 spenders

filt=df['Item Name'].value_counts().nlargest(5)
print(filt)

print(filt.index)
#Popular 5 
filt=df['Item Name'].isin(filt.index)

df[filt]['Price'].sum()  # purchase value for popular 5

df[filt]['Item ID'].unique() # Item ID of top Most

#purchase count
df[filt]["Purchase ID"].count()

