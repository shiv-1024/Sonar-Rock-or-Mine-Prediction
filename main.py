import pandas as pd  
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

sonar_df = pd.read_csv(r"D:\Files\Machine Learning\Projects\Sonar Rock Mine Prediction\sonar data.csv",header = None)

print(f'First five line of Dataset:\n{sonar_df.head()}')

print(f'Shape of the Sonar Data {sonar_df.shape}')

print(f'Statistical Data of Sonar Data:\n{sonar_df.describe()}')

print(f'The Outcomes Values :{sonar_df.value_counts(60)}')

print(f'The Mean of the Value in Outcomme:\n{sonar_df.groupby(60).mean()}')

x = sonar_df.drop(columns=60, axis = 1)
y = sonar_df[60]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y ,random_state = 1 )

print(f'The shape of the training data and testing data :{x_train.shape,x_test.shape}')

#Model Training
model = LogisticRegression()
model.fit(x_train,y_train)

#Model Evalution
x_train_prediction =  model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prediction,y_train)

print(f'The accuracy of the model on the training data is {x_train_accuracy}')

input_data = (0.0331,0.0423,0.0474,0.0818,0.0835,0.0756,0.0374,0.0961,0.0548,0.0193,0.0897,0.1734,0.1936,0.2803,0.3313,0.5020,0.6360,0.7096,0.8333,0.8730,0.8073,0.7507,0.7526,0.7298,0.6177,0.4946,0.4531,0.4099,0.4540,0.4124,0.3139,0.3194,0.3692,0.3776,0.4469,0.4777,0.4716,0.4664,0.3893,0.4255,0.4064,0.3712,0.3863,0.2802,0.1283,0.1117,0.1303,0.0787,0.0436,0.0224,0.0133,0.0078,0.0174,0.0176,0.0038,0.0129,0.0066,0.0044,0.0134,0.0092)
input_data_into_array = np.asarray(input_data)
input_data_reshape = input_data_into_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)

if prediction[0] == 'R':
    print('The Object is Rock')
else:
    print('The Object is Mine')

