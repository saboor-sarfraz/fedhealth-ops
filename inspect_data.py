
import numpy as np

files = {
    "Hospital A": "data/hospital_A.npz",
    "Hospital B": "data/hospital_B.npz",
    "Hospital C": "data/hospital_C.npz",
    "Global Test": "data/global_test.npz"
}

for hospital, file_path in files.items():

    print("\n" + "=" * 50)
    print(hospital)
    print("=" * 50)

    data = np.load(file_path, allow_pickle=False)

    print("Available arrays:", data.files)

    for key in data.files:

        array = data[key]

        print(f"\nArray: {key}")
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("First few values:")
        print(array.flatten()[:10])
