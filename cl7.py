import numpy as np

def generate_dummy_data(samples=100, features=10):
    """Generate random data and binary labels."""
    data = np.random.rand(samples, features)
    labels = np.random.randint(0, 2, size=samples)
    return data, labels

class AIRS:
    def __init__(self, num_detectors=5):
        """Initialize AIRS with a specified number of detectors."""
        self.num_detectors = num_detectors
        self.detectors = None

    def train(self, X):
        """Select a subset of data as detectors randomly."""
        indices = np.random.choice(len(X), self.num_detectors, replace=False)
        self.detectors = X[indices]

    def predict(self, X):
        """Predict labels based on nearest detector using Euclidean distance."""
        predictions = []
        for sample in X:
            distances = np.linalg.norm(self.detectors - sample, axis=1)
            prediction = np.argmin(distances)
            predictions.append(prediction)
        return predictions

# Create dummy data and split it
data, labels = generate_dummy_data(samples=50, features=5)  # Smaller dataset for simplicity
split_index = int(len(data) * 0.8)
train_data, test_data = data[:split_index], data[split_index:]

# Initialize, train, and test AIRS
airs = AIRS(num_detectors=3)
airs.train(train_data)
predictions = airs.predict(test_data)

# Calculate and print accuracy
accuracy = np.mean(predictions == labels[split_index:])
print(f"Accuracy: {accuracy:.2f}")