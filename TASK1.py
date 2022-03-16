#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

#Data Reading 
Hours=[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8]
Scores=[21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]

#finding correlation
np.corrcoef(Scores,Hours)

#Importing linear regreesion model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#converting the data into array
Hours=np.array(Hours)
Scores=np.array(Scores)

#reshaping the data to perform training and testing
Hours1=Hours.reshape(-1,1)
scores1=Scores.reshape(-1,1)

#traing and testing
Hours1_train,Hours1_test,scores1_train,scores1_test=train_test_split(Hours1,scores1,test_size=0.2)
model=LinearRegression().fit(Hours1_train,scores1_train)
model.coef_,model.intercept_
line=model.coef_*Hours1+model.intercept_

#plotting the graph
plt.xlabel("Hours1",fontsize=15)
plt.ylabel("scores1",fontsize=15)
plt.scatter(Hours1,scores1)
plt.show()

#Predicting the score if a student studies for 9.25 hours a day?
predictions=model.predict(Hours1_test)
predictions
(scores1_test,predictions)
pred_result=model.predict(np.array([[9.25]]))
pred_result
print('Mean Absolute Error:',metrics.mean_absolute_error(scores1_test,predictions))
