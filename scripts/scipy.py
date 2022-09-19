#### terminal commands ####

'pip install -r requirements.txt'




#### imports ####

import pandas as pd
import pandas.plotting as pdp ## the scipy lecture asks you to input 'from pandas.tools import plotting' here. depending on the version of pandas installed, you may have to run this instead
import numpy as np
import researchpy as rp
from scipy import stats
import urllib
import os
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm




#### Data representation and interaction ####

data = pd.read_csv(
    'data/brain_size.csv', sep=';', na_values='.'
    )

data

## here we simply use pandas to import the example data provided by the scipy lecture as we have done many times before

t = np.linspace(
    -6, 6, 20
    )

sin_t = np.sin(t)

cos_t = np.cos(t)

pd.DataFrame(
    {'t':t, 'sin':sin_t, 'cos':cos_t}
    )

## here we pivot and use numpy to create arrays and lists and then using pandas again to build a dataframe

data.shape

data.columns

print(
    data['Gender']
    )

## simple commands that allow us to visualize the dataset

data[data['Gender'] == 'Female']['VIQ'].mean()

## after we viewed the Gender column, we get more specific by selecting Female as the desired value of the Gender and taking the mean of the VIQ values connected to those rows

data.describe()

## pd describe is another method for a quick description of datasets

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print(
        (gender, value.mean())
        )

## instead of taking the VIQ means of just Female Gender rows, we can do the same with Male rows and put them side by side for comparison with .groupby

groupby_gender.mean()

## in fact, by just applying .mean() to the function we defined with .groupby, we get a table of mean values across all columns with each Gender


#### Exercise ####

## what is the mean value of VIQ for the whole population?

data['VIQ'].mean()

## we select the VIQ column from the dataset and apply .mean() to it
## 112.35

## how many males/females were included in this study?

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['Gender']:
    print(
        gender, value.count()
        )

## we can repeat the same code used in the mean VIQ by Gender example, except this time we change value to Gender and apply .count() instead of .mean()
## 'Female', 20
## 'Male', 20

## what is the average value of MRI counts expressed in log units, for males and females?

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['MRI_Count']:
    print(
        gender, np.log(value.mean())
        )

## similar concept, but we adjust value to MRI_Count and apply numpy's log to the mean of this value.
## Female 13.6678
## Male 13.7693

pdp.scatter_matrix(
    data[['Weight', 'Height', 'MRI_Count']]
    )

pdp.scatter_matrix(
    data[['PIQ', 'VIQ', 'FSIQ']]
    )

## here we plot scatter matrices for each of the columns in groups of three
## both matrices are bimodal

pdp.scatter_matrix(
    data[data['Gender'] == 'Female']
    )

pdp.scatter_matrix(
    data[data['Gender'] == 'Male']
    )

## we can plot matrices here for Male and Females separately to investigate whether the sub populations are related to Gender or not




#### Hypothesis testing: comparing two groups ####

stats.ttest_1samp(
    data['VIQ'], 0
    )

## a 1-sample t test on VIQ with all VIQ data

female_viq = data[data['Gender'] == 'Female']['VIQ']

male_viq = data[data['Gender'] == 'Male']['VIQ']

stats.ttest_ind(
    female_viq, male_viq
    )

## this time we separate and define two VIQ samples based on Gender and run a 2-sample test

stats.ttest_ind(
    data['FSIQ'], data['PIQ']
    )

## here we are testing FSIQ and PIQ are significantly different with stats.ttest_ind which we used earlier

stats.ttest_rel(
    data['FSIQ'], data['PIQ']
    )  

## variables that are measured on the same individuals like the IQ measurements are not appropriate to use ttest_ind on. It's more appropriate to use .ttest_rel which means a "paired test". this addresses variance from inter-subject variability.

stats.ttest_1samp(
    data['FSIQ'] - data['PIQ'], 0
    ) 

## another measure we could take is simply taking a 1-test sample on the difference between FSIQ and PIQ

stats.wilcoxon(
    data['FSIQ'], data['PIQ']
    )   

## the Wilcoxon test does not assume Gaussian errors. it is a type of non parametric test
## scipy.stats.mannwhitneyu() is the equivalent of the Wilcoxon for non-paired variables.
## it is not appropriate to use here


#### Exercise ####

## test the difference between weights in males and females

female_weight = data[data['Gender'] == 'Female']['Weight']

male_weight = data[data['Gender'] == 'Male']['Weight']

stats.ttest_ind(female_weight, male_weight)

## use non parametric statistics to test the difference between VIQ in males and females

stats.wilcoxon(
    data[data['Gender'] == 'Male']['VIQ'], data[data['Gender'] == 'Female']['VIQ']
)




#### Linear models, multiple factors, and analysis of variance ####

## simple linear regression

## for two sets of observations x and y, with y being the dependent, we want to test the hypothesis that y is a linear function of x

## we are going to fit a linear model using ordinary least squares (OLS) with statsmodels

x = np.linspace(-5, 5, 20)

np.random.seed(1)

## generating stimulated data according to model
## normal distributed noise

y = -5 + 3*x + 4 * np.random.normal(size=x.shape)

## create a df containing all the relevant variables

data = pd.DataFrame({'x': x, 'y': y})

from statsmodels.formula.api import ols

model = ols("y ~ x", data).fit()

print(model.summary()) ## let's inspect the stats from the ols fit


## categorical variables

data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".") ## we will reuse the example data

model = ols("VIQ ~ Gender + 1", data).fit() 

print(model.summary()) ## let's compare IQ in males and females as a linear model and view it 

model = ols('VIQ ~ C(Gender)', data).fit() ## if we want to force a numerical variable to be treated as a categorical variable, we can use this
## in this way, a numerical column like VIQ can be treated the same as a categorical one like Gender

## Now we want to compare IQ types by creating a long-form table by listing IQs and their type

data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})

data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})

data_long = pd.concat((data_fisq, data_piq))

print(data_long)

stats.ttest_ind(data['FSIQ'], data['PIQ']) ## let's double check our results by comparing the p values we got for the earlier t test


## multiple regression

## this time we consider a model using z as the dependent variable and x and y as the independents
## we'll use dataset iris.csv

data = pd.read_csv('examples/iris.csv')

model = ols('sepal_width ~ name + petal_length', data).fit()

print(model.summary())


## ANOVA testing

print(model.f_test([0, 1, -1, 0]))




#### Seaborn! ####

if not os.path.exists('wages.txt'):

    ## Download the example data file

    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', ## scipy lecture asks you to run .urlretrieve, we run .request.urlretrieve here
                       'data/wages.txt')

## Give names to the columns

names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]

short_names = [n.split(':')[0] for n in names]

data = pd.read_csv('data/wages.txt', skiprows=27, skipfooter=6, sep=None,
                       header=None)

data.columns = short_names

## Log-transform the wages, because they typically are increased with multiplicative factors

data['WAGE'] = np.log10(data['WAGE'])

## pairplot: scatter matrices

sb.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                      kind='reg')

sb.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                      kind='reg', hue='SEX')
plt.suptitle('Effect of gender: 1=Female, 0=Male')

sb.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                      kind='reg', hue='RACE')
plt.suptitle('Effect of race: 1=Other, 2=Hispanic, 3=White')

sb.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                        kind='reg', hue='UNION')

plt.suptitle('Effect of union: 1=Union member, 0=Not union member')

## simple regression plot 
## a regression capturing the relation between one variable and another, eg wage and eduction, can be plotted using sb.lmplot():

sb.lmplot(y='WAGE', x='EDUCATION', data=data)

plt.show() ## view data on multiple plots

print(data) ## view data in traditional table format

from matplotlib import pyplot as plt
plt.rcdefaults() ## if for some reason a user prefers trad matplotlib styling instead of more modern seaborn styling, this code will do the trick on the version revert


## to use a robust model, which is less sensitive to outliers, we can specify robust=True in seaborn plotting functions
## or we can replace OLS with robust linear model (statsmodels.formula.api.rlm()) in statsmodels




#### Testing for interactions #### 

## we have plotted various different fits. if we now want to formulate a single model to test for variance across slopes between two populations, we use interactions

result = sm.OLS(formula='wage ~ education + gender + education * gender', data=data).fit() ## OLS asks for parameters endog and exdog

print(result.summary())