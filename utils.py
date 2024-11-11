import numpy as np
import h5py
import pandas as pd
from scipy.signal import butter, filtfilt

def extract_data_from_hdf5(file_path):
    """
    Extract all variables and convert them into a dictionary format.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: Dictionary containing all data.
    """

    with h5py.File(file_path, 'r') as hdf5_file:
        ClassA = [hdf5_file[cell_data[0]][()].astype(np.uint16).tobytes().decode('utf-16')
                  for cell_data in hdf5_file['ClassA']]

        ClassS = [hdf5_file[cell_data[0]][()].astype(np.uint16).tobytes().decode('utf-16')
                  for cell_data in hdf5_file['ClassS']]

        ECG = [hdf5_file[cell_data[0]][()].flatten()
               for cell_data in hdf5_file['ECG']]

        SpO2 = [hdf5_file[cell_data[0]][()].flatten()
                for cell_data in hdf5_file['SpO2']]

        EEG = [hdf5_file[cell_data[0]][()].flatten()
               for cell_data in hdf5_file['EEG']]

        QRS = [hdf5_file[cell_data[0]][()].flatten()
               for cell_data in hdf5_file['QRS']]

        SR_ECG = hdf5_file['SR_ECG'][()][0, 0]
        SR_EEG = hdf5_file['SR_EEG'][()][0, 0]
        SR_SpO2 = hdf5_file['SR_SpO2'][()][0, 0]

    return {
        'ClassA': ClassA,
        'ClassS': ClassS,
        'ECG': ECG,
        'SpO2': SpO2,
        'EEG': EEG,
        'QRS': QRS,
        'SR_ECG': SR_ECG,
        'SR_EEG': SR_EEG,
        'SR_SpO2': SR_SpO2
    }


def segment_data(signals, label, signal_type="ECG"):
    """
    Segment signal data and pair each segment with the corresponding label.

    Parameters:
    signals (list of list of floats): List of input signal data, where each element is a patient's signal.
        - Each signal is a list of floats representing the signal intensity at sampling points.
        - For example, signals[i] is the signal of the i-th patient, and signals[i][j] is the signal value
          of the i-th patient at the j-th sampling point.

    label (list of list of int): List of corresponding label data, where each element is a patient's label.
        - Each patient's label is a list of integers representing the class label of each signal segment.
        - For example, label[i] is the label of the i-th patient, and label[i][j] is the class label
          of the j-th segment of the patient's signal.

    signal_type (str, optional): Type of the signal, default is "ECG".
        - "ECG": Electrocardiogram signal
        - "EEG": Electroencephalogram signal
        - "SpO2": Blood oxygen signal
        - Different signal types use different processing methods (filtering and artifact removal).

    Returns:
    segments (list of numpy.ndarray): Segmented signals, each element is a numpy array representing a segment.
        - The length of each signal segment is determined by signal_type, with a default of 6000 sampling points (ECG).
        - Zero-padding is applied if the signal length is insufficient.

    numeric_labels (list of int): Flattened list of labels representing the class label for each segment.
        - Label 'N' is mapped to 0, 'H' to 1, and 'A' to 2.
    """
    if signal_type == "SpO2":
        segment_length = 30
    else:
        segment_length = 6000
    labels = []
    segments = []
    for signal, class_a in zip(signals, label):
        num_segments = len(class_a)  # The number of segments for each signal is equal to the number of class labels

        # Process each segment
        for j in range(num_segments):
            start_idx = j * segment_length
            end_idx = min((j + 1) * segment_length, len(signal))  # Prevent index out of bounds

            # If the signal length is sufficient, copy directly; otherwise, pad with zeros
            if end_idx <= len(signal):
                segment = signal[start_idx:end_idx]
            else:
                segment = np.zeros(segment_length)
                segment = signal[start_idx:segment_length]
                segment[0:end_idx - start_idx] = signal[start_idx:end_idx]

            if signal_type == "ECG":
                segment = bandpass_filter(segment, lowcut=0.5, highcut=40, fs=200)
            #     hrv
            elif signal_type == "EEG":
                segment = bandpass_filter(segment, lowcut=0.5, highcut=32, fs=200)
            elif signal_type == "SpO2":
                segment = moving_average(segment, window_size=5)
                segment = remove_artifacts(segment)
            else:
                raise ValueError("Unsupported signal type. Please choose 'ECG', 'EEG', or 'SpO2'.")
            segments.append(segment)

        labels.extend(class_a)
    label_mapping = {'N': 0, 'H': 1, 'A': 2}
    numeric_labels = [label_mapping[label] for label in labels]

    print("label_mapping = {'N': 0, 'H': 1, 'A': 2}")
    return segments, numeric_labels


# Define the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Calculate the Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Moving average filter
def moving_average(signal, window_size):
    target_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    target_signal[0:2] = signal[0:2]
    target_signal[-2:] = signal[-2:]
    return target_signal


def remove_artifacts(spo2_signal, min_val=70, max_val=100):
    # Mark values as NaN or interpolate if they are out of the reasonable range
    cleaned_signal = np.where((spo2_signal < min_val) | (spo2_signal > max_val), np.nan, spo2_signal)

    # Interpolate to replace NaN values
    cleaned_signal = pd.Series(cleaned_signal).interpolate(method='linear').to_numpy()
    return cleaned_signal



#Five-fold Cross Validation for XGboost
model = xgb.XGBClassifier(eval_metric='mlogloss')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f" {fold + 1} fold")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = xgb.XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  
    
    print(f"acc: {acc}")
    print(f"F1 : {f1}\n")

#Five-fold Cross Validation for Deep Learning Model
for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset, labels)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # class_weights = torch.tensor([1.00, 5.00, 10.00]).to(device)
    model = SENet_LSTM(num_classes=2)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    
    num_epochs = 30
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
    
        for data, labels1 in train_loader:
            data = data.to(device)
            labels1 = labels1.to(device)
           
            outputs = model(data)
            loss = criterion(outputs, labels1)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
    
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels1.size(0)
            correct_train += (predicted == labels1).sum().item()
    
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    
   
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_samples, val_labels in val_loader:
           
            val_samples = val_samples.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_samples)
            _, predicted = torch.max(outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()

    fold_accuracy = correct / total
    fold_accuracies.append(fold_accuracy)
    print(f'Accuracy for fold {fold+1}: {fold_accuracy * 100:.2f}%\n')


average_accuracy = sum(fold_accuracies) / k_folds
print(f'Average K-Fold Cross Validation Accuracy: {average_accuracy * 100:.2f}%')
