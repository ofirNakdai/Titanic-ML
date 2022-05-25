# Name of student #1:
# ID of  student #1:

# Name of student #2:
# ID of  student #2:


# Download the "Titanic - Machine Learning from Disaster" dataset, either from Kaggle (registration required) or from the course's website
# https://www.kaggle.com/competitions/titanic/
# To quote from the website:
# "This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML"
# The goal of this exercise is to explore the dataset a bit, get some insights about important attributes of survivors and build an end-to-end machine learning model which predicts the survival of passengers on board the Titanic.

# In this exercise, we first explore the data, and then try to predict the survival odds.
# You are free to use as many loops as you like, and any library functions from numpy, pandas and sklearn, etc... 
# But you should remember, that if you are iterating over the rows of a DataFrame - most probably you're doing something wrong, not to mention it is going to be VERY slow.

# You should implement the body of each function. The functions are invoked in the correct order in the supplied main section.
# The main function implements an end-to-end classification pipeline including mainly a data EDA and prediction parts.
# Note that you should fill your code in the placeholders denoted by "<your code here>" and/or  #! your code here. It does NOT imply a single-line solution
# Make sure that you didn't miss any code filling part by making sure that main() function runs with no Exceptions.
# You are free to change the body of the functions as you like as long as it performs the required printings and returns the expected variables. 
# In all functions the required returned variables are explicitly stated.

# Final submission instruction: in addition to stating your names and ID numbers in the body of this file, name the file in the following way:
#
# ex3_FirstName1_LastName1_FirstName2_LastName2.py
#
# where FirstName1, ... stand, naturally, for your name(s)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()




# 1. 
# Familiarization. Download the dataset from the course's website to your pc / colab, and load it

## 1.1. 
## Load train.csv
def load_train_data():
    df_train = pd.read_csv("./train.csv")
    return df_train

## 1.2. 
## Display some data - display the top 10 rows
def disp_some_data(df_train):
    print(df_train.head(n=10))
    return

## 1.3. 
## In order to know what to do with which columns, we must know what types are there, and how many different values are there for each.
def display_column_data(df_train, max_vals = 10):
    
    '''
    First, let's investigate the columns in the dataset columns, using the .info() command. For now, we'll deal just with int, float and strings.
    Note that "object" is for string
    '''

    print(df_train.info())    
    '''
    Let's count the number of unique values per column:
    '''
    num_uq_vals_sr = df_train.nunique()
    print(num_uq_vals_sr)

    '''
    num_uq_vals_sr is pandas Series. It's index are the column names - "Survived", "Pclass", "Name", "Sex", ...
    Its values are the number of unique values for each column. For "Sex", for example, it is 2. For "Pclass", it is 3 - 3 classes of passengers.
    It discards NaN's by default. You can count NaN's as well with: df_train.nunique(dropna=False). Try it!
    '''

    '''
    For columns that have less than max_vals values, print the number of occurrences of each value
    '''
    #! your code here
    columns_to_print = num_uq_vals_sr[num_uq_vals_sr.values < max_vals].index

    for col in columns_to_print:
        print('{:s}: '.format(col), dict(df_train[col].value_counts()))
        
    
    return


## 1.4
## Now that we know which columns are there, we can drop some of them - the ones that do not carry predictive power.
## In addition we will drop columns that we do not know how to handle such as free text.
## Drop the columns: PassengerId, Name, Ticket, Cabin
def drop_non_inform_columns(df_train):

    #! your code here, store your results in df_lean
    df_lean = df_train.drop(labels= ["PassengerId" , "Name", "Ticket", "Cabin"], axis= 1)

    return df_lean



# 2. 
# Now that we know the basics about our dataset, we can start cleaning & transforming it towards enabling prediction of survival

## 2.1 
## In which columns are there missing values?
def where_are_the_nans(df_lean):

    #! your code here: print and return the names of the columns that have at least one missing value, and the number of missing values
    #! store your results in a dict or a series, where the index/key is the column name, and the value is the number of nans. For example:
    #!
    #! cols_with_nans = {'col_1': 20, 'col_5': 1, 'col_4': 13}
    #! DO NOT include columns with 0 NaN's - columns without missing values.
    cols_with_nans = df_lean.isna().sum()
    
    for col in cols_with_nans.keys():
        if cols_with_nans[col] == 0:
           cols_with_nans = cols_with_nans.drop(labels=col)
    
    cols_with_nans = cols_with_nans.to_dict()
    print(cols_with_nans)

    return cols_with_nans

## 2.2
## We see that the columns 'Age' and 'Embarked' have missing values. We need to fill them. 
## Let's fill 'Age' with the average and 'Embarked' with the most common
def fill_titanic_nas(df_lean):

    '''
    For "Embarked", consider using value_counts() to get (again) the value counts, 
    and idxmax() on that result - to get the index in of the maximal value value_counts() - that is most common value in "Embarked"
    '''

    #! your code here
    
    Embarked_Fill = df_lean.value_counts("Embarked").idxmax()
    Age_Fill = (df_lean["Age"].sum())/len(df_lean["Age"])


    df_filled = df_lean[:]
    df_filled["Embarked"] = df_filled["Embarked"].fillna(Embarked_Fill)
    df_filled["Age"] = df_filled["Age"].fillna(Age_Fill)

    return df_filled

## 2.3
## Now that we filled up all the missing values, we want to convert the non-numerical (categorical) variables 
## to some numeric representation - so we can apply numerical schemes to it.
## We'll encode "Embarked" and "Pclass", using the "one-hot" method
def encode(df_filled):

    '''
    There are 3 distinct values for "Embarked": "S", "C", "Q". Also, there are 3 classes of tickets. While the column "Pclass" is numeric, 
    it is more categorical than numeric - the number of the class does not bear special meaning, and could've easily been A, B, C, and not 1, 2, 3.
    We shall encode these two variables by the "one-hot" scheme, which produces numerical values for categorical variables (columns).

    For a categorical variable X, with 3 values x1, x2, x3, "one-hot" introduces 3 new binary variables - X_1, X_2, X_3, where each represents whether X is valued
    x1, x2 or x3. It is called "one-hot", because at each row, only a single variable of X_1, X_2, X_3 will be 1, and the rest will be 0.

    In short: for a categorical variable with, say 7 distinct values, "one-hot" introduces 7 new binary variables, 
    where at each row exactly one of the new variables is 1, with the rest being zero


    For example, suppose we have the following "Embarked" column:
 
     Index | "Embarked" | ...  
    ------ +------------+------
        0  |   "S"      | ...  
        1  |   "C"      | ...  
        2  |   "C"      | ...  
        3  |   "Q"      | ...  
        4  |   "S"      | ...  
        5  |   "Q"      | ...  
 

    Then, applying "one-hot" to it, and naming the new columns "Emb_C", "Emb_Q", "Emb_S", will yield the following table:
   

     Index | "Embarked" | "Emb_C" | "Emb_Q" | "Emb_S" | ...   
    -------+------------+---------+---------+---------+------
        0  |   "S"      | 0       | 0       | 1       | ...       
        1  |   "C"      | 1       | 0       | 0       | ...    
        2  |   "C"      | 1       | 0       | 0       | ...    
        3  |   "Q"      | 0       | 1       | 0       | ...    
        4  |   "S"      | 0       | 0       | 1       | ...    
        5  |   "Q"      | 0       | 1       | 0       | ...       



    Applying "one-hot" to "Embarked" introduces 3 new columns, where for each of the new columns we'll place 1 if the passenger embarked from that port, and 0 otherwise. 

    '''

    # Apply one-hot to "Embarked" and "Pclass" columns. 
    # For "Embarked", the new coumns should be named "Emb_C", "Emb_Q", "Emb_S"
    # For "Pclass", the new coumns should be named "Cls_1", "Cls_2", "Cls_3"

    # ****************************
    # ***** !!!!! ALSO !!!!! *****
    # ****************************
    #
    # For lack of better place, introduce a new column, "Bin_Sex" - a binary (1 or 0) version of the "Sex" column
    #
    # ****************************
    # ***** !!!!! ALSO !!!!! *****
    # ****************************

    


    #! your code here. Hint: you are strongly encouraged to use pd.get_dummies(...) function and then rename the columns. 
    df = df_filled.copy()
    df["Emb_C"] = 0
    df["Emb_Q"] = 0
    df["Emb_S"] = 0
    df["Emb_C"] = df["Emb_C"].where(df["Embarked"] != 'C',1)
    df["Emb_Q"] = df["Emb_Q"].where(df["Embarked"] != 'Q',1)
    df["Emb_S"] = df["Emb_S"].where(df["Embarked"] != 'S',1)    

    df["Cls_1"] = 0
    df["Cls_2"] = 0
    df["Cls_3"] = 0
<<<<<<< HEAD
    df["Cls_1"] = df["Cls_1"].where(df["Pclass"] != 1,1)
    df["Cls_2"] = df["Cls_2"].where(df["Pclass"] != 2,1)
    df["Cls_3"] = df["Cls_3"].where(df["Pclass"] != 3,1)     
=======
    df["Cls_1"] = df["Cls_1"].where(df["Pclass"] != '1',1)
    df["Cls_2"] = df["Cls_2"].where(df["Pclass"] != '2',1)
    df["Cls_3"] = df["Cls_3"].where(df["Pclass"] != '3',1)     
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5

    df["Bin_Sex"] = 0
    df["Bin_Sex"] = df["Bin_Sex"].where(df["Sex"] != "male",1)
    
    
    df = df.drop(labels=["Embarked", "Pclass", "Sex"], axis=1)
     # *** NOTE ***: after encoding by one-hot, we may delete the original columns, although it is not necessery. 
    

    return df


## 2.4 
## There are 2 variables (columns) that reflect co-travelling family of each passenger. 
## SibSp - the number of sibling - brothers and sisters.
## Parch - the total number of parents plus children for each passneger.
## We want to reflect the whole family size of each passenger - the sum of SibSp and Parch
## It will be useful later
def make_family(df):

    '''
    Introduce a new column with the name "Family", that will be the sum of "SibSp" and "Parch" columns
    '''

    #! your code here
    df['Family'] = df["SibSp"] + df["Parch"]

    return df


## 2.5 Feature Transformation
## In many cases, it is the *multiplicative* change in some numeric variable that affects the outcome, not the *additive* one.
## For example, we expect that the change of survival probability of 16 year olds relative to 12 year olds is *much greater*
## than the change of survival probability of 48 year olds relative to 44 year olds. 
## To capture that notion, we take the log of the variables of interest. To guard against taking log of 0, we add 1 prior to that.
## All in all, it produces not bad results. 
## In short: X -> log(1+X)
## There is a numpy function exactly for that: np.log1p
## This will be useful later
def add_log1p(df):

    # For each of the numeric columns: 'Age', 'SibSp', 'Parch', 'Fare', 'Family'
    # we introduce a new column that starts with the 'log1p_' string: 'log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare', 'log1p_Family'

    for col in ['Age', 'SibSp', 'Parch', 'Fare', 'Family']:
        df['log1p_' + col] = np.log1p(df[col])

    return df


# 3. Basic exploration of survival. 
# This section deals with correlations of the "Survived" column to various other data about the passengers.
# Also, in this section, we can still use the df_filled DataFrame, without "one-hot" encoding. It is up to you.
## 3.1. Survival vs gender
def survival_vs_gender(df):
    '''
    What is the survival rate for women and men?
    '''

    #! Compute the survival rate of women and men. That is, compute the percentage of survived women and survived men.
    #! Gender is specified in the "Sex" column.
    #! You should - but do not have to - do it by defining a view of the DataFrame that includes, for example, only men, and then computing the average of "Survived"
    #! For example:
    #! df_male = df[df['Sex']=='male']
    #! Now, compute the average of "Survived":
    #! df_male["Survived"].mean()

    #! your code here
    #! Return the result in a dict or a series. Your keys for dict / index for Series should be the strings "male", "female"
    df_male = df[df["Bin_Sex"]== 1 ] #using df
    df_female = df[df["Bin_Sex"]== 0 ]
    survived_by_gender = {"male": df_male["Survived"].mean(), "female": df_female["Survived"].mean()}

    print(survived_by_gender)
    return survived_by_gender


## 3.2 The same for survival by class. You can use the "one-hot" encoding, or the original "Pclass" column - whatever more convenient to you.
def survival_vs_class(df):

    #! your code here
    #! Return the result in a dict or a series. Your keys for dict / index for Series should be the strings "Cls_1", "Cls_2", "Cls_3"
    # For instance: survived_by_class = {"Cls_1": .25, "Cls_2": .35, "Cls_3": .45}
    
    df_Cls_1 = df[df["Cls_1"]== 1 ] #using df
    df_Cls_2 = df[df["Cls_2"]== 1 ] #using df
    df_Cls_3 = df[df["Cls_3"]== 1 ] #using df

    survived_by_class = {"Cls_1": df_Cls_1["Survived"].mean(), "Cls_2": df_Cls_2["Survived"].mean(), "Cls_3": df_Cls_3["Survived"].mean()}
    print(survived_by_class)

    return survived_by_class


## 3.3 The same, for survival by the three family size metrics. Return a dict of dicts / series
def survival_vs_family(df):
    
    '''
    The different family size metrics - "SibSp", "Parch", "Family" are all numeric. 
    '''
    
    survived_by_family = {}

    for metric in ["SibSp", "Parch", "Family"]:

        #! your code here
        survived_by_metric = { }  
        for x in df[metric].unique():
            survived_by_metric[x] = (df[df[metric]== x ])["Survived"].mean()         
        # Example for survived_by_metric:
        # survived_by_metric = {0: 0.2, 1: 0.3, 2: 0.35, 4: 0.5...}
        # here the keys are unique values of each metric, and the values are the survival probability.
        # So in this example, we got 4 unique values for the metric - 0,1,2,4 , and the corresponding survival probabilities are 0.2, 0.3, 0.35, 0.5
        
        print("Family metric: ", metric)
        print("Survival stats:")
        print(survived_by_metric)
        
        survived_by_family[metric] = survived_by_metric
        
      #! What survival metric with what value ensures the highest probability of survival?
      #! Complete the following print statement after inspecting the reuslts
      
      #! your code here
<<<<<<< HEAD
    best_metric = " "
    highest = 0
    best_val = -1
    for metric in survived_by_family.keys():
        find_max = max(survived_by_family[metric], key=survived_by_family[metric].get)
        if(survived_by_family[metric][find_max] > highest):
            highest = survived_by_family[metric][find_max]
            best_metric = metric
            best_val = find_max
    print("To ensure the highest chance of survival, the metric ", best_metric,'must have the value ', best_val )      
    
=======
        best_metric = " "
        highest = 0
        for metric in survived_by_family.keys():
            find_max = max(survived_by_family[metric], key=survived_by_family[metric].get)
            if(survived_by_family[metric][find_max] > highest):
                highest = survived_by_family[metric][find_max]
                best_metric = metric
        print("To ensure the highest chance of survival, the metric ", best_metric,'must have the value ', highest )      
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5
    return survived_by_family


## 3.4 Visualizing the distribution of age and its impact on survival 
def survival_vs_age(df):
    '''
    Here we would like to plot some histograms. 
    Some very basic plotting: run these three commands:

    plt.plot(np.arange(10))
    plt.plot(np.arange(10)**.5)
    plt.plot(np.arange(10)**2)
    
    You should get 3 lines on a single figure. While that is a desirable functionality, things can quickly go out of hand if you do not close or clear your figures.

    To prevent clogging your figures with clutter, you can do one of several things:

    1. put:
    plt.close('all')
    at the beginning of this function. This will close all figures, all subsequent plots will be "new"

    2. Naming your figures and closing / clearing them before the first plot in the function.

    2.1. Closing:

    plt.close('abc') # closes figure "abc", no error raised if it does not exist.
    plt.figure('abc') # opens a brand, empy figure named "abc"
    plt.plot(...)
    df[...].hist()

    2.1. clearing:

    Here the order is reversed

    plt.figure('abc') # makes "abc" the active figure, makes a new figure named "abc" if it does not exist
    plt.clear('abc') # clears the figure "abc", leaving it empty and active to recieve plots
    plt.plot(...)
    df[...].hist()

    
    Now, back to age histogram (distributions)
    First, define the histogram edges

    The following is a suggestion, the choice is up to you (will not affect the grade except for a really bad case, like bins = [0,100])
    '''
    bins = list(range(0,100,4))
    '''
    We can plot a histogram of any column of numerical values by the "hist" method of DataFrame
    '''

    plt.close('Age, all')
    plt.figure('Age, all')
    df['Age'].hist(bins=bins)
    plt.close('Age, all')
    


    '''
    Note, you can also put: bins = 'auto'
    df['Age'].hist(bins='auto')

    Try it! 

    '''

    #! your code here
    #! plot 2 histograms of age: one for those who survived, and one for those that did not
    #! Bonus 1: plot 4 histograms of age: for women that survived and not, and for men tat survived and not
    #! Bonus 2: plot 6 histograms of age: for survivors and casualties of each of the 3 classes
    
    Age_Survived = df["Age"].where(df["Survived"] == 1)
    Age_Not_Survived = df["Age"].where(df["Survived"] == 0)

    women_Survived = df["Age"].where(df["Survived"] == 1).where(df["Bin_Sex"] == 0)
    men_Survived = df["Age"].where(df["Survived"] == 1).where(df["Bin_Sex"] == 1)
    women_Not_Survived = df["Age"].where(df["Survived"] == 0).where(df["Bin_Sex"] == 0)
    men_Not_Survived = df["Age"].where(df["Survived"] == 0).where(df["Bin_Sex"] == 1)

    plt.figure("Age_Survived")
    plt.title("Survived by Age")
    plt.xlabel("Age")
    plt.ylabel("Survivors")
    Age_Survived.hist(bins='auto')
    plt.savefig("Age_Survived")
    plt.close("Age_Survived")

    plt.figure("Age_Not_Survived")
    plt.title('Not Survived by Age')
    plt.xlabel("Age")
    plt.ylabel("Not Survived")
    Age_Not_Survived.hist(bins='auto')
    plt.savefig("Age_Not_Survived")
    plt.close("Age_Not_Survived")

    plt.figure("women_Survived")
    plt.title('Women Survived by Age')
    plt.xlabel("Age")
    plt.ylabel("Survived")
    women_Survived.hist(bins='auto')
    plt.savefig("women_Survived")
    plt.close("women_Survived")

    plt.figure("men_Survived")
    plt.title('Men Survived by Age')
    plt.xlabel("Age")
    plt.ylabel("Survived")
    men_Survived.hist(bins='auto')
    plt.savefig("men_Survived")
    plt.close("men_Survived")

    plt.figure("women_Not_Survived")
    plt.title('Women Not Survived by Age')
    plt.xlabel("Age")
    plt.ylabel("Not Survived")
    women_Not_Survived.hist(bins='auto')
    plt.savefig("women_Not_Survived")
    plt.close("women_Not_Survived")

    plt.figure("men_Not_Survived")
    plt.title('Men Not Survived by Age')
    plt.xlabel("Age")
    plt.ylabel("Not Survived")
    men_Not_Survived.hist(bins='auto')
    plt.savefig("men_Not_Survived")
    plt.close("men_Not_Survived")
        
    
    return


## 3.5 Correlation of survival to the numerical variables
# ['Age', 'SibSp', 'Parch', 'Fare', 'Family']
# ['log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare', 'log1p_Family']
def survival_correlations(df):

    '''
    We can compute the correlation of the various numeric columns to survival. This is done by the corr function of DataFrame.

    '''

    corr = df.corr()

    # corr is a DataFrame that represents the correlatio matrix
    print(corr)

    # we need only the correlation to the "Survived" column. Also, we don't need the correlation of "Survived" to itself.
    # Also, remember, that for inspection purposes, it is the *absolute value* of correlation that's important

    #! your code here
    #! find the 5 most important numerical columns, and print (with sign) and return their correlation. Use dict or Series

<<<<<<< HEAD
    important_feats = corr["Survived"][1:].abs().sort_values(ascending=False).head().index

    important_corrs = corr["Survived"][important_feats]

=======
    important_feats = corr["Survived"]
    important_feats.drop(labels=["Survived"], inplace=True)
    abs_important_feats = important_feats.abs().sort_values(ascending=False).dropna()
    importent_corrs = {}
    
    for i in range(5):
        importent_corrs[abs_important_feats.keys()[i]] = important_feats[abs_important_feats.keys()[i]]
    
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5
    print(important_corrs)

    return important_corrs



# 4. Predicting survival!!!
# We're finally ready to build a model and predict survival! 
# 
# In this section, df should include all the transformations and additions we've done to the data, including, of course, the one-hot
# encoding of class and port of boarding (Embarked), and the binary "Sex" column, and also with the addition of the log1p scaled variables.
# But including too much features, not to metntion redundant ones (think log1p_Age and Age), can deteriorate the performance. 
# Based on the correlations of the numeric data to survival, and the impact of the categorical data, you're encouraged to pick the best features 
# that will yield the best testing results.    
        

## 4.1 split data into train and test sets
def split_data(df):



    from sklearn.model_selection import train_test_split

    Y = df['Survived']
    X = df.drop(['Survived'], axis = 1)



    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=1, stratify = Y)

    # This splits the data into a train set, which will be used to calibrate the internal parameters of predictor, and the test set, which will be used for checking

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test


## 4.2 Training and testing!
def train_logistic_regression(X_train, X_test, y_train, y_test):


    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import f1_score 
=======
    from sklearn.metrics import confusion_matrix
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5

    para_grid = {'C' : [0.001, 0.01, 0.1, 1, 10 ,50], # internal regularization parameter of LogisticRegression
                 'solver' : ['sag', 'saga']}

    Logit1 = GridSearchCV(LogisticRegression(penalty='l2' ,random_state=1), para_grid, cv = 5)

    Logit1.fit(X_train, y_train)

    y_test_logistic = Logit1.predict(X_test)

    '''
    look at:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    to interpret y_test_logistic
    
    
    Now let's see how good our model is. Compute, print and return the following three performance measures, as taught in class:
    
    1. Confusion matrix
    2. Accuracy
    3. F1 score
    
    For that, in sklearn.metrics look at:
    1. sklearn.metrics.confusion_matrix
    2. sklearn.metrics.accuracy_score
    3. sklearn.metrics.f1_score
    
   
    '''

    #! your code here
<<<<<<< HEAD
    conf_matrix =  confusion_matrix(y_test, y_test_logistic)
    accuracy =  accuracy_score(y_test, y_test_logistic)
    f1_score =  f1_score(y_test, y_test_logistic)
=======
    conf_matrix =  confusion_matrix(X_test, y_test_logistic)
    accuracy =  <your code here>
    f1_score =  <your code here>
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5

    print('acc: ', accuracy, 'f1: ', f1_score)
    print('confusion matrix:\n', conf_matrix)
    
    return accuracy, f1_score, conf_matrix



    
    
    


if __name__ == '__main__':
  
    # an example of the usage of the functions

    df_train = load_train_data()
    disp_some_data(df_train)
    display_column_data(df_train, max_vals = 10)
    df_lean = drop_non_inform_columns(df_train)
    
    cols_with_nans = where_are_the_nans(df_lean)
    df_filled = fill_titanic_nas(df_lean)
    df = encode(df_filled)
    df = make_family(df)
    df = add_log1p(df)
 
    survived_by_gender = survival_vs_gender(df)
    survived_by_class = survival_vs_class(df)
    survived_by_family = survival_vs_family(df)
    survival_vs_age(df)
    important_corrs = survival_correlations(df)
    
<<<<<<< HEAD
    final_df = df.drop(labels=['Age', 'SibSp', 'Parch', 'Fare', 'Family','log1p_SibSp', 'log1p_Age'], axis=1)   
    X_train, X_test, y_train, y_test = split_data(final_df)
=======
    X_train, X_test, y_train, y_test = split_data(df)
>>>>>>> 7e1bbb2ed59e89faa2b72d89200b3edb411dc7f5
    train_logistic_regression(X_train, X_test, y_train, y_test)

    






