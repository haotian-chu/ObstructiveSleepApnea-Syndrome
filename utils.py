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
