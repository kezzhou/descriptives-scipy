#### terminal commands ####

'pip install -r requirements.txt'




#### imports ####

import pandas as pd
import pandas.plotting as pdp ## the scipy lecture asks you to input 'from pandas.tools import plotting' here. depending on the version of pandas installed, you may have to run this instead
import numpy as np
import researchpy as rp
from scipy import stats




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