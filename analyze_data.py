
import numpy as np

# Load global test dataset
data = np.load("data/global_test.npz")

X = data["x"]
y = data["y"]

print("GLOBAL TEST ANALYSIS")
print("--------------------")

print("X shape:", X.shape)
print("Y shape:", y.shape)

# Check label distribution
labels, counts = np.unique(y, return_counts=True)

print("\nLabel distribution:")

for label, count in zip(labels, counts):
    print(f"Label {label}: {count} samples")

# Check signal statistics
print("\nSignal statistics:")
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Standard deviation:", X.std())

# Check hospital training data
for hospital in ["a", "b", "c"]:

    data = np.load(f"data/hospital_{hospital}.npz")

    X_train = data["x_train"]
    X_val = data["x_val"]

    print(f"\nHOSPITAL {hospital.upper()}")
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    print("Training mean:", X_train.mean())
    print("Training std:", X_train.std())

    print("Validation mean:", X_val.mean())
    print("Validation std:", X_val.std())
