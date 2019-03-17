import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# load data
home_data =pd.read_csv('train.csv')

## step 1: specify prediction target 
home_data.columns # print list of columns
y = home_data.SalePrice

### step 2: create features
feature_names =['LotArea','YearBuilt','1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr','TotRmsAbvGrd']
X  = home_data[feature_names]

print(X.describe())


###step 3: Specify and Fit Model

# step 3a: split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state =1)
# step 3b: define model
iowa_rf_model = RandomForestRegressor(random_state=1)
# step 3c: fit model
iowa_rf_model.fit(train_X,train_y)


###step 4: predict
rf_predict = iowa_rf_model.predict(val_X)


###step 5: validate check prediction
rf_val_mae = mean_absolute_error(rf_predict, val_y)
print("Validation MAE for random forest model: {}".format(rf_val_mae))



import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


cla()   # Clear axis
clf()   # Clear figure
close() # Close a figure window

scatter_matrix(X)

plt.subplot(211)
plt.scatter(train_X['LotArea'], train_y, s=1)
plt.subplot(212)
plt.scatter(train_X['YearBuilt'], train_y, s=1)
