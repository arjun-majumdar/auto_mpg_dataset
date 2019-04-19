

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector


'''
Regression Task
Target attribute- 'mpg'

https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
'''


# Read in data file-
data = pd.read_csv("auto-mpg.data", delim_whitespace = True)

# Get dimension/shape of dataset-
data.shape
# (398, 9)

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

# Get attribute name(s) having missing values-
data.isnull().sum()
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

data_scaled = rbs.fit_transform(data_filled_na)		# fit and transform

# Convert from np.ndarray to pandas DataFrame-
data_scaled = pd.DataFrame(data_scaled, columns=data_filled_na.columns)


# Get features (X) and label (y) from dataset-
X = data_scaled.drop('mpg', axis = 1)
y = data_scaled['mpg']


# Divide features (X) and label (y) into training and testing sets-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print("\nDimensions of training and testing sets are:\n")
print("X_train = {0}, y_train = {1}, X_test = {2} & y_test = {3}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# Dimensions of training and testing sets are:
# X_train = (277, 8), y_train = (277,), X_test = (120, 8) & y_test = (120,)




# Use different regression models-

# Initialize a Linear Regression model-
lr_model = LinearRegression()

# Train model on training data-
lr_model.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = lr_model.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nLinear Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Linear Regression model metrics:
# MSE = 0.0864, MAE = 0.2242 & R2-Score = 0.8059

# Use 5-fold cross-validation-
cvs = cross_val_score(lr_model, X_train, y_train, cv = 5)

print("\nLinear Regression Cross Validation (5-fold CV) are:")
print("Mean = {0:.4f} & Standard deviation = {1:.4f}\n".format(cvs.mean(), cvs.std()))
# Linear Regression Cross Validation (5-fold CV) are:
# Mean = 0.8033 & Standard deviation = 0.0436


# Remove highly correlated features to see whether new model is equivalent to as compared
# to using all attributes-

new_cols = X_train.columns.tolist()

# 'cylinder' attribute is highly correlated to attributes- 'displacement', 'horsepower'
# and 'weight'

# new_cols.remove('displacement')
new_cols.remove('horsepower')
new_cols.remove('weight')


# Instantiate a new Linear Regression model-
new_lr_model = LinearRegression()

# Train on training data-
new_lr_model.fit(X_train.loc[:, new_cols], y_train)

# Make predictions using trained model-
y_pred_new = new_lr_model.predict(X_test.loc[:, new_cols])


# Get model metrics-
mse = mean_squared_error(y_test, y_pred_new)
mae = mean_absolute_error(y_test, y_pred_new)
r2s = r2_score(y_test, y_pred_new)

print("\n\nLinear Regression model metrics using less attributes:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Linear Regression model metrics using less attributes:
# MSE = 0.1308, MAE = 0.2704 & R2-Score = 0.7441
# MSE = 0.1281, MAE = 0.2640 & R2-Score = 0.7492




# Initialize a Lasso Regression model-
lasso_model = Lasso()

# Train model on training data-
lasso_model.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = lasso_model.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nLasso Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Lasso Regression model metrics:
# MSE = 0.4493, MAE = 0.5370 & R2-Score = -0.0095




# Initialize a Ridghe Regression model-
ridge_model = Ridge()

# Train model on training data-
ridge_model.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = ridge_model.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nRidge Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Ridge Regression model metrics:
# MSE = 0.0861, MAE = 0.2216 & R2-Score = 0.8065




# Use Decision Tree Regressor-

# Initialize DT regressor-
dt_reg = DecisionTreeRegressor()

# Train model on training data-
dt_reg.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = dt_reg.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nDecision Tree Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Decision Tree Regression model metrics:
# MSE = 0.1053, MAE = 0.2228 & R2-Score = 0.7635




# Use XGBoost regressor-
xgb_reg = xgb.XGBRegressor()

# Train model on training data-
xgb_reg.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = xgb_reg.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nXGBoost Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# XGBoost Regression model metrics:
# MSE = 0.0597, MAE = 0.1768 & R2-Score = 0.8659

cvs = cross_val_score(xgb_reg, X_train, y_train, cv = 5)

print("\nXGBoost Cross Validation (5-fold CV) are:")
print("Mean = {0:.4f} & Standard deviation = {1:.4f}\n".format(cvs.mean(), cvs.std()))
# XGBoost Cross Validation (5-fold CV) are:
# Mean = 0.8477 & Standard deviation = 0.0434




# Use LightGBM regressor-
lgb_reg = lgb.LGBMRegressor()

# Train model on training data-
lgb_reg.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = lgb_reg.predict(X_test)


# Get model metrics-
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print("\n\nLightGBM Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# LightGBM Regression model metrics:
# MSE = 0.0594, MAE = 0.1789 & R2-Score = 0.8666

# Use 5-fold cross-validation-
cvs = cross_val_score(lgb_reg, X_train, y_train, cv = 5)

print("\nLightGBM Cross Validation (5-fold CV) are:")
print("Mean = {0:.4f} & Standard deviation = {1:.4f}\n".format(cvs.mean(), cvs.std()))
# LightGBM Cross Validation (5-fold CV) are:
# Mean = 0.8565 & Standard deviation = 0.0601




# Use Random Forest regressor-

# Initialize a RF regressor-
rf_reg = RandomForestRegressor(n_estimators=140)

# Train model on training data-
rf_reg.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_rf = rf_reg.predict(X_test)

# Get model metrics-
mse = mean_squared_error(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)
r2s = r2_score(y_test, y_pred_rf)

print("\n\nRandom Forest Regression model metrics:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# Random Forest Regression model metrics:
# MSE = 0.0630, MAE = 0.1803 & R2-Score = 0.8735

# Use 5-fold cross-validation-
cvs = cross_val_score(rf_reg, X_train, y_train, cv = 5)

print("\nRandom Forest regressor Cross Validation (5-fold CV) are:")
print("Mean = {0:.4f} & Standard deviation = {1:.4f}\n".format(cvs.mean(), cvs.std()))
# Random Forest regressor Cross Validation (5-fold CV) are:
# Mean = 0.8417 & Standard deviation = 0.0885




# Perform Recursive Feature Elimination-

# Dictionary to save: number_attributes: r2_score-
n_params_rfe = {}

# Dictionary to save: number_attributes: name_attributes
name_attributes_rfe = {}

for n_cols in range(2, X_train.shape[1] + 1):

	print("\nCurrent # of parameters being used = {0}".format(n_cols))

	# Use a LightGBM regressor for RFE using 'i' parameters-
	lgb_reg = lgb.LGBMRegressor()

	rfe_lgb_reg = RFE(lgb_reg, n_cols)

	# Train RFE object on features (X) and label (y)-
	rfe_lgb_reg = rfe_lgb_reg.fit(X_train, y_train)

	# To get names of selected/extracted features/attributes-
	attributes_selected = X_train.columns[rfe_lgb_reg.support_].tolist()

	name_attributes_rfe[n_cols] = attributes_selected
	
	# Now train a new LightGBM regressor using these 'n_cols' attributes to compute
	# R-Squared score-
	lgb_reg_new = lgb.LGBMRegressor()
	
	# Train new LightGBM regressor on training data-
	lgb_reg_new.fit(X_train.loc[:, attributes_selected], y_train)

	# Make predictions using trained model-
	y_pred = lgb_reg_new.predict(X_test.loc[:, attributes_selected])

	# Get R-Squared score of trained model-
	r2s = r2_score(y_test, y_pred)

	n_params_rfe[n_cols] = r2s



# Visualize accuracy vs. number of parameters used to train RF classifier-
plt.plot(n_params_rfe.keys(), n_params_rfe.values())

plt.xlabel("Number of parameters used")
plt.ylabel("R2-Score")
plt.title("Feature Selection - Recursive Feature Elimination using LightGBM Regressor")
plt.show()


'''
n_params_rfe

{2: 0.6821638134355141,
 3: 0.8733048320443797,
 4: 0.8742505932748215,
 5: 0.8687547629155739,
 6: 0.8727488709860716,
 7: 0.8738794413186645,
 8: 0.8741301540704822}
'''

# number of attributes giving highest R2-Score is = 4
# name of attributes for 4 number of attributes-
name_attributes_rfe[4]
# ['weight', 'acceleration', 'model year', 'car_name_encoded']


# Finally, instantiate a new LightGBM regressor-
new_lgb_reg = lgb.LGBMRegressor()

# Train model on trainingn data using the specified attributes-
new_lgb_reg.fit(X_train.loc[:, name_attributes_rfe[4]], y_train)

# Make predictions using trained model-
y_pred_new = new_lgb_reg.predict(X_test.loc[:, name_attributes_rfe[4]])

# Get model metrics-
mse = mean_squared_error(y_test, y_pred_new)
mae = mean_absolute_error(y_test, y_pred_new)
r2s = r2_score(y_test, y_pred_new)

print("\n\nLightGBM Regression model metrics using specified attributes:")
print("MSE = {0:.4f}, MAE = {1:.4f} & R2-Score = {2:.4f}\n\n".format(mse, mae, r2s))
# LightGBM Regression model metrics using specified attributes:
# MSE = 0.0643, MAE = 0.1958 & R2-Score = 0.8743

# Get 5-fold CV score-
cvs = cross_val_score(new_lgb_reg, X_train.loc[:, name_attributes_rfe[4]], y_train, cv = 5)
print("\nLightGBM regressor 5-fold CV score:")
print("Mean = {0:.4f} & Standard Deviation = {1:.4f}\n\n".format(cvs.mean(), cvs.std()))
# LightGBM regressor 5-fold CV score:
# Mean = 0.8469 & Standard Deviation = 0.0359




# Performing Forward Feature Selection-
# A dictionary to store: number_parameters: mean_squared_error
# n_param_mse_ffs = {}		# ffs -> forward feature selection
n_param_r2_ffs = {}

# Dictionary to save: number_attributes: name_attributes
name_attributes_ffs = {}

for i in range(2, X_train.shape[1] + 1):

	print("\nCurrent number of parameters = {0}\n".format(i))

	# feature_selector = SequentialFeatureSelector(lgb.LGBMRegressor(), k_features = i, forward = True, scoring = 'neg_mean_squared_error', cv = 5)
	feature_selector = SequentialFeatureSelector(lgb.LGBMRegressor(), k_features = i, forward = True, scoring = 'r2', cv = 5)

	# Train defined feature selector on training data-
	features_selected = feature_selector.fit(np.array(X_train.fillna(0)), y_train) # use this

	# See the attributes/features selected-
	feature_selected_op = X_train.columns[list(features_selected.k_feature_idx_)].tolist()

	name_attributes_ffs[i] = feature_selected_op

	# Train an XGBoost classifier-
	lgb_reg = lgb.LGBMRegressor()
	lgb_reg.fit(X_train.loc[:, feature_selected_op], y_train)
	
	# Make predictions using trained model-
	y_pred = lgb_reg.predict(X_test.loc[:, feature_selected_op])

	# Get base model metrics-
	# mse =  mean_squared_error(y_test, y_pred)
	r2s = r2_score(y_test, y_pred)

	# n_param_mse_ffs[i] = mse
	n_param_r2_ffs[i] = r2s


# Visualize accuracy vs. number of parameters used to train RF classifier-
plt.plot(n_param_r2_ffs.keys(), n_param_r2_ffs.values())

plt.xlabel("Number of parameters used")
plt.ylabel("R2-Score")
plt.title("Feature Selection - Forward Feature Selection using LightGBM Regressor")
plt.show()


'''
# Get number_attributes: r2_score-
n_param_r2_ffs

{2: 0.8680169300036188,
 3: 0.8644449941579877,
 4: 0.8751112742138695,
 5: 0.8753069314195098,
 6: 0.8747404540487316,
 7: 0.8738794413186645,
 8: 0.8741301540704822}
'''

# Get name of attributes for 5 attributes giving highest R2-Score
name_attributes_ffs[5]
# ['displacement', 'horsepower', 'weight', 'model year', 'origin']


# Observation:
# In this case, Forward Feature Selection gives slightly higher R2-Score as compared to
# Recursive Feature Selection
# FFS gives higher R2-Score for 5 parameters whereas RFE gives slightly R2-Score for 4
# parameters




# Use a Neural Network regressor-
nn_reg = Sequential()

# Input Layer-
nn_reg.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))

# Output layer-
nn_reg.add(Dense(1, kernel_initializer = 'normal'))