# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 04:27:42 2022

@author: Lion PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dexplot as dxp


pd.set_option('display.max_columns', None)

# import the CSV file into df 
df=pd.read_csv('travel_insurance.csv')

#display the deails of the data
print(df.info())

#display the first 5 rows 
print(df.head())

    
"""
customers are those who bough insurance marked as 1 in TravelInsurance column

Customer Question
They want to know:
● Are there differences in the travel habits between customers and non-customers?
● What is the typical profile of customers and non-customers?

we are going to check the other charactristics grouping by TravelInsurance column
and in the end we are going to use a simple classifier 

preform data tidyness steps:
    1- the comlumn names should be clearer:
        Age > age
        Employment Type > employment_type,
        GraduateOrNot > graduate,
        AnnualIncome > annual_income,
        FamilyMembers > family_members,
        ChronicDiseases > chronic_diseases,
        FrequentFlyer > frequent_flyer,
        EverTravelledAbroad' > ever_travelled_abroad,
        TravelInsurance > travel_insurance
    2- the yes/no columns to 1&0:
        graduate
        chronic_diseases
        frequent_flyer
        ever_travelled_abroad
    3- change employment_type to Categorical type 
    4-check for missing values and outliers in age and annual_income
     


steps:
    1- divide the data into customers and non-customers datasets
    2- find patterns in both dataset to profile each type:
        1- for employment type
    3- find common habits among the subgroups of each set
    

"""
#renaming the columns
df.rename(columns = {'Age':'age','Employment Type':'employment_type', 'GraduateOrNot':'graduate',
                     'AnnualIncome':'annual_income', 'FamilyMembers':'family_members',
                     'ChronicDiseases':'chronic_diseases', 'FrequentFlyer':'frequent_flyer',
                     'EverTravelledAbroad':'ever_travelled_abroad', 'TravelInsurance':'travel_insurance'},
          inplace = True)


#change yes/no to 1&0
for col in ['graduate', 'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad']:
    df[col].replace({'Yes':1,'No':0},inplace=True)
    df[col]=df[col].astype(bool)

#change employment_type to categorical
df.employment_type = df.employment_type.astype('category')

#checking for any missing values
df.isnull().sum()

#checking annual_income and age outliers using boxplot
box_plot=sns.boxplot(df.age)
age_median=df.age.median()
plt.title('Age distribution')
plt.xlabel('Age')
box_plot.text(age_median,0.01,str(age_median)+' Years', fontsize=8,color='w')
plt.show()


box_plot=sns.boxplot(df.annual_income/1000000)
income_median=df.annual_income.median()/1000000
plt.title('Annual Income in Millions')
plt.xlabel('Annual Income in Millions')
box_plot.text(income_median,0.01,str(income_median)+' Million', fontsize=10,color='w')
plt.show()



 """
 create a class column and then plot the customers based on that
 """
  
#creating a function to classify the data based on income 
def class_label (row):
   if row['annual_income']>0.75*df.annual_income.max() :
      return 'high'
   if (row['annual_income']<0.75*df.annual_income.max())&(row['annual_income']>0.5*df.annual_income.max()):
      return 'middle'
   if row['annual_income']<0.5*df.annual_income.max():
      return 'low'
# apply the function to create the new feature
df['income_class']=df.apply (lambda row: class_label(row), axis=1)


#plot the insurance purchasing based on class
# sns.histplot(df, x="income_class", hue="travel_insurance", stat="percent", multiple="dodge", shrink=.8)
sns.catplot(data=df, x="income_class", hue="travel_insurance", kind="count",legend_out=False)


#using the emplyment type to classify the data
# sns.histplot(df, x="employment_type", hue="travel_insurance", stat="percent", multiple="dodge", shrink=.8)
sns.catplot(data=df, x="employment_type", hue="travel_insurance", kind="count",legend_out=False)


#comparing the income class and employment_type on insurance purchasing
g=sns.catplot(data=df, x="income_class", hue="travel_insurance",col='employment_type', kind="count",legend_out=False)



#creating a function to classify the data based on family_members 
def family_label (row):
   if row['family_members']>=8 :
      return 'huge'
   if (row['family_members']<8)&(row['family_members']>=6):
      return 'large'
   if (row['family_members']<6)&(row['family_members']>=4):
      return 'mid'
   if row['family_members']<4:
      return 'small'
# apply the function to create the new feature
df['family_size']=df.apply (lambda row: family_label(row), axis=1)

#comparing the income class and family_size on insurance purchasing
sns.catplot(data=df, x="family_size", hue="travel_insurance",col='income_class',
            order=['small','mid','large','huge'],kind="count",legend_out=False)



#==============================================================================
"""
to answer Are there differences in the travel habits between customers and non-customers?
we need to devide the set into customer and non customer groups based on travel_insurance column
then we can see how the mojority behave 
1- the family size
2- travel abroad
3- average age
4- employment type
5- graduation status
6- annual income
7- if they have chronic disease
8- if they are frequent flyer
"""

df.info()

# using travel_insurance to devide the data
df_customer=df[df.travel_insurance==1].reset_index(drop=True)
df_non_customer=df[df.travel_insurance==0].reset_index(drop=True)

#graduation VS purchasing
dxp.count('graduate', data=df,normalize=True,split='travel_insurance')

# Customers behover

box_plot=sns.boxplot(df_customer.age)
age_median=df_customer.age.median()
plt.title('cusomer Age')
box_plot.text(age_median,0.01,str(age_median)+' Years', fontsize=8,color='w')
plt.show()

dxp.count('age', data=df_customer,normalize=True, title='Cusomer age')
dxp.count('graduate', data=df_customer,normalize=True, title='Cusomer graduates')
dxp.count('employment_type', data=df_customer,normalize=True, title='Cusomer Employment')
dxp.count('chronic_diseases', data=df_customer,normalize=True, title='Cusomer Diseases')
dxp.count('ever_travelled_abroad', data=df_customer,normalize=True, title='Cusomer travelled abroad')






g=sns.catplot(data=df_customer, x="income_class", hue='employment_type', kind="count",legend_out=False)

g=sns.catplot(data=df_customer, x="graduate",  kind="count",legend_out=False)

dxp.count('graduate', data=df_customer,normalize=True, title='Cusomer graduates')
dxp.count('ever_travelled_abroad', data=df_customer,normalize=True, title='Cusomer travelled abroad')
dxp.count('chronic_diseases', data=df_customer,normalize=True, title='Cusomer Diseases')
dxp.count('employment_type', data=df_customer,normalize=True, title='Cusomer Employment')
dxp.count('frequent_flyer', data=df_customer,normalize=True, title='Cusomer frequent_flyer')


dxp.count('frequent_flyer', data=df_customer,split='graduate',normalize=True, title='graduate frequent flyer')
dxp.count('ever_travelled_abroad', data=df_customer,split='graduate',normalize=True, title='graduate travelled abraod')
dxp.count('chronic_diseases', data=df_customer,split='graduate',normalize=True, title='graduate chronic disease')

dxp.count('age', data=df_customer,split='ever_travelled_abroad',normalize=True, title='Age travelled abraode')


# Non-Customers behover

box_plot=sns.boxplot(df_non_customer.age)
age_median=df_non_customer.age.median()
plt.title('Non cusomer Age')
box_plot.text(age_median,0.01,str(age_median)+' Years', fontsize=8,color='w')
plt.show()


g=sns.catplot(data=df_non_customer, x="income_class", hue='employment_type', kind="count",legend_out=False)

dxp.count('age', data=df_non_customer,normalize=True, title='Cusomer age')
dxp.count('graduate', data=df_non_customer,normalize=True, title='Non Cusomer graduates')
dxp.count('ever_travelled_abroad', data=df_non_customer,normalize=True, title='Non Cusomer travelled abroad')
dxp.count('chronic_diseases', data=df_non_customer,normalize=True, title='Non Cusomer Diseases')
dxp.count('employment_type', data=df_non_customer,normalize=True, title='Non Cusomer Employment')
dxp.count('frequent_flyer', data=df_non_customer,normalize=True, title='Non Cusomer frequent_flyer')

dxp.count('frequent_flyer', data=df_non_customer,split='graduate',normalize=True, title='graduate frequent flyer')
dxp.count('ever_travelled_abroad', data=df_non_customer,split='graduate',normalize=True, title='graduate travelled abraod')
dxp.count('chronic_diseases', data=df_non_customer,split='graduate',normalize=True, title='graduate chronic disease')
dxp.count('age', data=df_non_customer,split='ever_travelled_abroad',normalize=True, title='Age travelled abraod')





dxp.count('age', data=df,split='travel_insurance',normalize=True, title='Age')
dxp.count('graduate', data=df,split='travel_insurance',normalize=True, title='graduate')
dxp.count('employment_type', data=df,split='travel_insurance',normalize=True, title='Employment')
dxp.count('chronic_diseases', data=df,split='travel_insurance',normalize=True, title='Diseases')
dxp.count('ever_travelled_abroad', data=df,split='travel_insurance',normalize=True, title='Employment')


dxp.count('age', data=df,split='ever_travelled_abroad',row='travel_insurance',normalize=True, title='Employment')
sns.catplot(data=df, x="age", hue='ever_travelled_abroad',col='travel_insurance', kind="count",legend_out=False)

dxp.count('family_members', data=df,split='travel_insurance',normalize=True, title='family Size')

dxp.count('family_size', data=df,split='travel_insurance',normalize=True, title='family Size')


dxp.count('income_class', data=df,split='travel_insurance',normalize=True, title='family Size', x_order='asc')



for col in ['age', 'employment_type', 'graduate', 'annual_income', 'family_members',
       'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad']:
    x=dxp.count(col, data=df_non_customer,split='travel_insurance',normalize=True, title='non customer '+col)
    display(x)



for col in ['age', 'employment_type', 'graduate', 'annual_income', 'family_members',
       'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad','income_class','family_size' ]:
    x=dxp.count(col, data=df_customer,normalize=True, title='customer '+col)
    display(x)
    x=dxp.count(col, data=df_non_customer,normalize=True, title='non customer '+col)
    display(x)



for col in ['age', 'employment_type', 'graduate', 'annual_income', 'family_members',
       'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad','income_class','family_size' ]:
    x=dxp.count(col, data=df,split='travel_insurance',normalize=True, title='customer '+col)
    display(x)
    
    
    




sns.set_theme(style="whitegrid")
sns.countplot(data=df_customer, x="age")

for col in ['age', 'employment_type', 'graduate', 'annual_income', 'family_members',
       'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad','income_class','family_size' ]:

    ax = sns.barplot(x=col, y=col, data=df_customer, estimator=lambda col: len(col) / len(df_customer) * 100)
    ax.set(ylabel="Percent")


sns.barplot(x='ever_travelled_abroad',  data=df_customer)

sns.barplot(x='employment_type', data=df_customer)





sns.barplot(x=df_customer.ever_travelled_abroad.value_counts().index, y=df_customer.ever_travelled_abroad.value_counts(normalize=True))
plt.ylabel('Percent')
plt.xlabel('Ever travelled abroad')
plt.title('Customers who travelled aboard')


fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Customer VS non-customer travelling aboard')
axes[0].set_title('Customers')
axes[1].set_title('Non-Customers')
sns.barplot(ax=axes[0],x=df_customer.ever_travelled_abroad.value_counts().index, y=df_customer.ever_travelled_abroad.value_counts(normalize=True))
sns.barplot(ax=axes[1],x=df_non_customer.ever_travelled_abroad.value_counts().index, y=df_non_customer.ever_travelled_abroad.value_counts(normalize=True))

axes[0].set_ylabel('percent')
axes[1].set_ylabel('')






fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Customer VS non-customer frequent flying')
axes[0].set_title('Customers')
axes[1].set_title('Non-Customers')
sns.barplot(ax=axes[0],x=df_customer.frequent_flyer.value_counts().index,
            y=df_customer.frequent_flyer.value_counts(normalize=True))
sns.barplot(ax=axes[1],x=df_non_customer.frequent_flyer.value_counts().index,
            y=df_non_customer.frequent_flyer.value_counts(normalize=True))

axes[0].set_ylabel('percent')
axes[1].set_ylabel('')





fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Customer VS non-customer frequent flying')
axes[0].set_title('Customers')
axes[1].set_title('Non-Customers')
sns.barplot(ax=axes[0],x=df_customer.chronic_diseases.value_counts().index,
            y=df_customer.chronic_diseases.value_counts(normalize=True))
sns.barplot(ax=axes[1],x=df_non_customer.chronic_diseases.value_counts().index,
            y=df_non_customer.chronic_diseases.value_counts(normalize=True))

axes[0].set_ylabel('percent')
axes[1].set_ylabel('')



fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Customer VS non-customer income class')
axes[0].set_title('Customers')
axes[1].set_title('Non-Customers')
sns.barplot(ax=axes[0],x=df_customer.income_class.value_counts().index,
            y=df_customer.income_class.value_counts(normalize=True))
sns.barplot(ax=axes[1],x=df_non_customer.income_class.value_counts().index,
            y=df_non_customer.income_class.value_counts(normalize=True))

axes[0].set_ylabel('percent')
axes[1].set_ylabel('')


fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Customer VS non-customer employment')
axes[0].set_title('Customers')
axes[1].set_title('Non-Customers')
sns.barplot(ax=axes[0],x=df_customer.employment_type.value_counts().index,
            y=df_customer.employment_type.value_counts(normalize=True))
sns.barplot(ax=axes[1],x=df_non_customer.employment_type.value_counts().index,
            y=df_non_customer.employment_type.value_counts(normalize=True))

axes[0].set_ylabel('percent')
axes[1].set_ylabel('')








sns.catplot(data=df, x="age", hue="income_class",col='employment_type',kind="count");


bins=15
sns.histplot(data=df, x=df.annual_income/1000000, bins=bins,hue='travel_insurance', multiple="dodge",stat="percent", common_norm=False,)
plt.title('Purchasing behavoir based on annual income')
plt.xlabel('Annual income in millions')



plt.xticks(np.round((np.linspace(df.annual_income.min(),df.annual_income.max()+100000,bins))/1000000,2))


np.round((np.linspace(df.annual_income.min(),df.annual_income.max()+100000,bins))/1000000,2)







#exporing the annual income
plt.figure(figsize=(15,10))
sns.histplot(data=df, x="age", bins=10,hue='travel_insurance', multiple="dodge")
plt.xticks(np.arange(df.age.min(),df.age.max(),1))

plt.xlabel('Age')
plt.title('Travels classification based on AGE');


sns.barplot(x=df.age.value_counts().index,
            y=df.age.value_counts(normalize=True))

plt.xlabel('Age')
plt.title('Travels classification based on AGE');

sns.countplot(data=df, x='age',hue='travel_insurance')
plt.xlabel('Age')
plt.title('Travels classification based on AGE');

['age', 'employment_type', 'graduate', 'annual_income', 'family_members',
       'chronic_diseases', 'frequent_flyer', 'ever_travelled_abroad','income_class','family_size' ]



x,y = 'annual_income', 'travel_insurance'


plt.figure(figsize=(15,10))
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, legend_out=False)
g.ax.set_ylim(0,100)

for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x-0.1,txt_y+1,txt, rotation=-90)

plt.xlabel('Age')
plt.title('Travels classification based on Age');




plt.figure(figsize=(5,5))
g=df.travel_insurance.value_counts(normalize=True).plot(kind='bar')
g.set_ylim(0,1)
for p in g.patches:
    txt = str((p.get_height()*100).round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.text(txt_x+0.1,txt_y+0.01,txt, rotation=0)
plt.title('Data classification')
plt.xticks([0,1], ['Non-Customer','Customer'],rotation=0)






#checking annual_income and age outliers using boxplot
box_plot=sns.boxplot(df.age)
age_median=df.age.median()
plt.xticks(np.arange(df.age.min(),df.age.max()+1,1))
plt.title('Age distribution')
plt.xlabel('Age')

box_plot.text(age_median,0.01,str(age_median)+' Years', fontsize=8,color='w')
plt.show()


box_plot=sns.boxplot(df.annual_income/1000000)
income_median=df.annual_income.median()/1000000
plt.title('Annual Income in Millions')
plt.xlabel('Annual Income in Millions')
box_plot.text(income_median,0.01,str(income_median)+' Million', fontsize=10,color='w')
plt.show()






