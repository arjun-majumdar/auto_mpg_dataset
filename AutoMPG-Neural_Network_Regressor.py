

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
'''
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
'''
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


'''
Regression Task
Target attribute- 'mpg'

https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
'''


# Read in data file-
# data = pd.read_csv("auto-mpg.data", delim_whitespace = True)
data = pd.read_csv("auto_mpg-processed_data.csv")

# Get dimension/shape of dataset-
data.shape
# (398, 9)

'''
# Assign attribute names to 'data'-
col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
	'model_year', 'origin', 'car_name']

# Assign column names to dataset-
data.columns = col_names

# Get attribute names (sanity check)-
data.columns.tolist()
'''
['mpg',
 'cylinders',
 'displacement',
 'horsepower',
 'weight',
 'acceleration',
 'model year',
 'origin',
 'car name']
'''

# Replace '?' characters with NaN values-
data = data.replace('?', np.nan)

# Check for missing values-
data.isnull().values.any()
# False
'''

# Get attribute name(s) having missing values-
# data.isnull().sum()
'''
mpg             0
cylinders       0
displacement    0
horsepower      6
weight          0
acceleration    0
model year      0
origin          0
car name        0
dtype: int64
'''

'''
# Convert 'horsepower' attribute from objet to numeric type-
data['horsepower'] = pd.to_numeric(data['horsepower'])

# Label Encoding for 'car name' attribute-

# Initialize a label encoder-
le = LabelEncoder()

# Train and encode-
car_name_encoded = le.fit_transform(data['car_name'])

# To reverse encoding-
# reverse_car_name = le.inverse_transform(car_name_encoded)

# Add attribute to dataset-
data['car_name_encoded'] = car_name_encoded

# Delete 'car name' attribute-
data.drop('car_name', axis =1 , inplace=True)


# Missing value(s) data imputation using 'fancyimpute' package-
data_filled_na = KNN(k = 3).fit_transform(data)

# Convert from np.ndarraty to pandas DataFrame-
data_filled_na = pd.DataFrame(data_filled_na, columns=data.columns)

# Write back to HDD-
# data_filled_na.to_csv("auto_mpg-processed_data.csv", index=False)
'''

'''
# Create a correlogram-

# Compute correlation matrix-
data_filled_na_corr = data_filled_na.corr()

# Create correlogram-
sns.heatmap(data=data_filled_na_corr)

plt.yticks(rotation = 20)
plt.xticks(rotation = 20)
plt.title("Auto MPG Dataset Heatmap")
plt.show()

# Observation-
# 'cylinders' attribute has a high correlation with-
# 'displacement', 'horsepower' & 'weight' attributes


# Visualize data distribution using boxplots-
sns.boxplot(data = data_filled_na)

plt.yticks(rotation = 20)
plt.xticks(rotation = 20)
plt.title("Auto MPG dataset distribution using Boxplots")
plt.show()
'''


# Normalize/scale data-
rbs = RobustScaler()	# Initialize scaler

# data_scaled = rbs.fit_transform(data_filled_na)		# fit and transform
data_scaled = rbs.fit_transform(data)		# fit and transform

# Convert from np.ndarray to pandas DataFrame-
# data_scaled = pd.DataFrame(data_scaled, columns=data_filled_na.columns)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)


# Get features (X) and label (y) from dataset-
X = data_scaled.drop('mpg', axis = 1)
y = data_scaled['mpg']


# Divide features (X) and label (y) into training and testing sets-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print("\nDimensions of training and testing sets are:\n")
print("X_train = {0}, y_train = {1}, X_test = {2} & y_test = {3}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# Dimensions of training and testing sets are:
# X_train = (277, 8), y_train = (277,), X_test = (120, 8) & y_test = (120,)






# Use a Neural Network regressor-

# First evalutate different NN topologies using Cross-Validation to find 'best' network-

def create_nn_regression_network():
	'''
	A function to create a neural network regressor
	'''
	
	# Create a NN regressor-
	nn_reg = Sequential()

	# Input Layer-
	nn_reg.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))

	# Hiddent Layer-
	nn_reg.add(Dense(8, kernel_initializer = 'normal', activation = 'relu'))

	# Output layer-
	nn_reg.add(Dense(1, kernel_initializer = 'normal'))

	# Compile NN model-
	nn_reg.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])

	# Return compiled network-
	return nn_reg


# Wrap Keras model so it can be used by scikit-learn module-
nn_regressor = KerasRegressor(build_fn = create_nn_regression_network,
		epochs = 600, batch_size = 5)


# Conduct k-fold cross-validation-
# Evaluate NN regression using 3-fold CV-
cvs_nn_reg = cross_val_score(nn_regressor, X_train, y_train, cv = 3)

print("\nNeural Network regressor 3-fold Cross-Validation results are:")
print("Mean = {0:.4f} & Standard Deviation = {1:.4f}\n\n".format(cvs_nn_reg.mean(), cvs_nn_reg.std()))
# Neural Network regressor 3-fold Cross-Validation results are:
# Mean = -0.0631 & Standard Deviation = 0.0108




# Finally, create the 'best' found NN-

# Create a NN regressor-
nn_reg = Sequential()

# Input Layer-
nn_reg.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))

# Hiddent Layer-
nn_reg.add(Dense(8, kernel_initializer = 'normal', activation = 'relu'))

# Output layer-
nn_reg.add(Dense(1, kernel_initializer = 'normal'))


# Compile NN model-
nn_reg.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Train our model-
history = nn_reg.fit(X_train, y_train, epochs = 600, batch_size = 5, validation_data = (X_test, y_test))


print(history.history.keys())
# dict_keys(['val_loss', 'loss'])

# Visualize the different losses-
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
plt.ylabel("Loss/Error")
plt.xlabel("# of epochs")
plt.title("NN regressor model loss")
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()

# Frpm the plot it seems that the model is overfitting to data. Reduce overfitting!


# Make predictions using trained NN regressor-
y_pred = nn_reg.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nNeural Network Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# No hidden layer-
# Neural Network Regression model metrics:
# MSE = 0.0737, MAE = 0.1972 & R2-Score = 0.8472

# One hidden layer-
# Neural Network Regression model metrics:
# MSE = 0.0546, MAE = 0.1673 & R2-Score = 0.8815
