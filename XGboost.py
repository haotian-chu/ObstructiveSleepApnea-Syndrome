from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
import xgboost as xgb


def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
    # Convert floating-point hyperparameters to integers or other appropriate ranges
    params = {
        'max_depth': int(max_depth),  # Depth of the trees
        'learning_rate': learning_rate,  # Learning rate
        'n_estimators': int(n_estimators),  # Number of trees
        'gamma': gamma,  # Minimum loss reduction required to make a further partition
        'min_child_weight': min_child_weight,  # Minimum sum of weights in child nodes
        'subsample': subsample,  # Proportion of samples used for each tree
        'colsample_bytree': colsample_bytree,  # Proportion of features used for each tree
        'objective': 'binary:logistic',  # Binary classification task
        'eval_metric': 'logloss'  # Loss function
    }

    # Train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Return accuracy as the optimization objective
    return accuracy_score(y_test, y_pred)


# Define the search space
pbounds = {
    'max_depth': (3, 10),  # Depth of the trees, range 3 to 10
    'learning_rate': (0.01, 0.3),  # Learning rate, range 0.01 to 0.3
    'n_estimators': (100, 300),  # Number of trees, range 100 to 300
    'gamma': (0, 5),  # Minimum loss reduction for node splitting
    'min_child_weight': (1, 10),  # Minimum sum of weights in child nodes
    'subsample': (0.6, 1),  # Proportion of samples
    'colsample_bytree': (0.6, 1)  # Proportion of features
}

# Initialize the Bayesian optimizer
optimizer = BayesianOptimization(
    f=xgb_evaluate,  # Target function
    pbounds=pbounds,  # Search space
    random_state=42,  # Fixed random seed
    verbose=2  # Display optimization process
)

# Start optimization, 10 iterations
optimizer.maximize(init_points=5, n_iter=10)

# Output the best parameter combination
print("Best parameters:", optimizer.max)
