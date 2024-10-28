import numpy as np
import pandas as pd

def generate_random_context(seq, max_length, stride, pad_token_id):
    """
    Generate sliding windows from a sequence with a given maximum length and stride.

    Args:
        seq (list): The input sequence of token IDs.
        max_length (int): Maximum length of each window.
        stride (int): Stride length between windows.
        pad_token_id (int): Token ID used for padding.

    Returns:
        list: A list of windows, each of which is a list of token IDs.
    """
    windows = []

    # If the sequence is shorter than or equal to max_length, pad and return a single window
    if len(seq) <= max_length:
        window = seq + [pad_token_id] * (max_length - len(seq))
        windows.append(window)
    else:
        # Create windows by moving through the sequence with the given stride
        for i in range(0, len(seq), stride):
            window = seq[i:i + max_length]
            # Pad the last window if it's shorter than max_length
            if len(window) < max_length:
                window += [pad_token_id] * (max_length - len(window))
            windows.append(window)
            # Stop if the window is shorter than max_length (end of sequence)
            if len(window) < max_length:
                break

    return windows


def create_random_context_matrix(barcodes, tokenized_sequences, max_length, stride, pad_token_id, seed=None):
    """
    Create a DataFrame mapping window IDs to cell barcodes.

    Args:
        barcodes (list): List of cell barcodes.
        tokenized_sequences (list): List of tokenized sequences for each cell.
        max_length (int): Maximum length of each window.
        stride (int): Stride length between windows.
        pad_token_id (int): Token ID used for padding.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing window IDs and corresponding cell barcodes.
    """
    if seed is not None:
        np.random.seed(seed)

    window_id_list = []
    barcode_list = []
    position_ids_list = []
    all_windows = []
    window_id_counter = 0

    for barcode, seq in zip(barcodes, tokenized_sequences):
        # Generate sliding windows for the current sequence
        windows = generate_random_context(seq, max_length, stride=stride, pad_token_id=pad_token_id)
        all_windows.extend(windows)
        for window in windows:
            # Create a unique window ID for each window in the sequence
            window_id = f"{barcode}_window_{window_id_counter}"
            window_id_counter += 1
            window_id_list.append(window_id)
            barcode_list.append(barcode)
            position_ids_list.append(list(range(len(window))))  # Position IDs for each token in the window

    # Create a DataFrame that maps window IDs to cell barcodes
    window_id_barcode_df = pd.DataFrame({
        'window_id': window_id_list,
        'cell_barcode': barcode_list
    })

    return window_id_barcode_df, all_windows


def prediction_random_context_matrix(barcodes, tokenized_sequences, max_length, pad_token_id, y_encoded, rank, targetname, seed=None):
    window_id_list = []
    barcode_list = []
    allwindows = []
    labels_list = []
    gpuid = str(rank)
    position_ids_list = []
    window_id_counter = 0
    stride = int(max_length/2)
    window_id_mapping = []
    if y_encoded is None:
        y_encoded = [None] * len(barcodes)
    for barcode, seq, label in zip(barcodes, tokenized_sequences, y_encoded):
        windows = generate_random_context(seq, max_length, stride, pad_token_id=pad_token_id)
        allwindows.extend(windows)
        barcode_list.extend([barcode] * len(windows))
        if targetname != "NOTROUBLESHOOT":
            labels_list.extend([label] * len(windows))  # Only add labels if not in "NOTROUBLESHOOT" mode
        for window in windows:
            window_id_str = f"{gpuid}_{barcode}_window_{window_id_counter}"
            window_id_mapping.append((window_id_counter, window_id_str))  # Store both numeric and string-based window IDs
            window_id_list.append(window_id_counter)
            window_id_counter += 1
            position_ids_list.append(list(range(len(window))))
    window_id_barcode_df = pd.DataFrame({
        'window_id': window_id_list,
        'cell_barcode': barcode_list,
    })
    if targetname != "NOTROUBLESHOOT":
        window_id_barcode_df['Label'] = labels_list
    return window_id_barcode_df, allwindows, labels_list, window_id_list, position_ids_list, window_id_mapping