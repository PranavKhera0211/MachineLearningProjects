import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(1)
n_samples, n_features = 400, 50
X = np.random.randn(n_samples, n_features)

# Generate weights with some sparsity
coef = 2 * np.random.randn(n_features)
coef[5:] = 0  # Only the first 5 features are relevant
y = X.dot(coef) + 0.1 * np.random.normal(size=n_samples)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Lasso regression (regularization = 0.1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Apply Ridge regression (regularization = 0.1)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print("Lasso Regression:")
print("Mean Squared Error:", mse_lasso)
print("Number of Nonzero Coefficients:", np.count_nonzero(lasso.coef_))

print("\nRidge Regression:")
print("Mean Squared Error:", mse_ridge)
print("Number of Nonzero Coefficients:", np.count_nonzero(ridge.coef_))

# Plot Lasso and Ridge coefficients
plt.subplot(1, 2, 1)
plt.bar(range(n_features), lasso.coef_)
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Lasso Regression Coefficients")

plt.subplot(1, 2, 2)
plt.bar(range(n_features), ridge.coef_)
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regression Coefficients")
plt.show()
