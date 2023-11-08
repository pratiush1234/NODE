import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


class NODEModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

model_wrapper = NODEModelWrapper(tabular_model)

# Calculate initial accuracy
initial_accuracy = accuracy_score(y_test, model_wrapper.predict(test)['prediction'].values)
print("Initial Accuracy:", initial_accuracy)

# Define a scoring function for accuracy
def accuracy_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred['prediction'])

# Calculate permutation feature importance
result = permutation_importance(
    model_wrapper,
    test,
    y_test,
    scoring=accuracy_scorer,
    n_repeats=30,
    random_state=42,
)

# Get the importance scores
importance_scores = result.importances_mean

# Map the importance scores to feature names
feature_importance = dict(zip(
    df_encoded.columns[::-1],
    importance_scores
))

# Sort the features by importance
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: -item[1]))

# Print feature importance
print("Feature Importance:")
for feature, importance in sorted_feature_importance.items():
    print(f"{feature}: {importance:.4f}")



import matplotlib.pyplot as plt


# Extract the feature names and importance scores
feature_names = list(sorted_feature_importance.keys())
importance_scores = list(sorted_feature_importance.values())

# Create a vertical bar graph
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title('Feature Importance')

# Display the plot
plt.show()




import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


scaling_factors = [1,2,3,4,5]

# Create lists to store the number of data points and the corresponding times
data_points = []
processing_times = []

# Loop through each scaling factor
for factor in scaling_factors:
    # Replicate the base dataset
    scaled_dataset = pd.concat([train] * factor, ignore_index=True)
    
    # Choose a specific operation (e.g., prediction) to measure time for
    operation_start_time = time.time()
    # Perform your operation on the scaled_dataset, e.g., prediction
    prediction = tabular_model.predict(scaled_dataset)
    operation_end_time = time.time()
    
    # Record the number of data points and the time taken for the operation
    data_points.append(len(scaled_dataset))
    processing_times.append(operation_end_time - operation_start_time)

# Plot the time vs. data points
plt.plot(data_points, processing_times)
plt.xlabel("Number of Data Points")
plt.ylabel("Processing Time (seconds)")
plt.title("Scaling Analysis")
plt.show()

