import numpy as np
import pandas as pd
import random
from ..utils.geneexpparser import *

    
def generate_genomic_context(seq, gene_symbol_file, max_length, depth, pad_token_id, seed, custom_tokenizer):
    np.random.seed(seed)
    reverse_gene_vocab = create_reverse_gene_vocab(custom_tokenizer)
    reverse_expression_vocab = create_reverse_expression_vocab(custom_tokenizer)    
    gene_symbol_df = pd.read_csv(gene_symbol_file, sep='\t')
    gene_cytoband_map = gene_symbol_df.set_index('gene_symbol')['cytobandID'].to_dict()
    windows = []
    cytoband_sequences = {}
    foundcyto = None
    paired_seq = [(seq[i], seq[i+1]) for i in range(0, len(seq), 2)]
    for gene_id, expr_id in paired_seq:
        gene = reverse_gene_vocab.get(gene_id, None)
        expr = reverse_expression_vocab.get(expr_id, None)
        cytoband = gene_cytoband_map.get(gene, None)
        
        if gene is None:
            raise ValueError(f"Found un-tracking-back gene {gene}")
        if expr is None:
            raise ValueError(f"Found un-tracking-back expression: gene {gene}, expression: {expr}")
            
        if cytoband:
            foundcyto = True
            if cytoband not in cytoband_sequences:
                cytoband_sequences[cytoband] = []
            cytoband_sequences[cytoband].append((gene_id, expr_id))
            
        else:
            if gene not in cytoband_sequences:
                cytoband_sequences[gene] = []
            cytoband_sequences[gene].append((gene_id, expr_id))
    cytobands = list(cytoband_sequences.keys())
    repeated_cytobands = cytobands * depth
    random.shuffle(repeated_cytobands)
    
    windows = []
    current_window = []
    for cytoband in repeated_cytobands:
        cytoband_seq = cytoband_sequences[cytoband]
        cytoband_seq = [item for pair in cytoband_seq for item in pair]
        if len(current_window) + len(cytoband_seq) <= max_length:
            current_window.extend(cytoband_seq)
        else:
            if len(current_window) < max_length:
                current_window += [pad_token_id] * (max_length - len(current_window))
            windows.append(current_window)  
            current_window = cytoband_seq
        
        if len(current_window) == max_length:
            windows.append(current_window)
            current_window = []

    if current_window:
        if len(current_window) < max_length:
            current_window += [pad_token_id] * (max_length - len(current_window))
        windows.append(current_window)

    if len(windows)==0:
        raise ValueError(f"ERROR: No window created")
    if foundcyto is None:
        raise ValueError(f"ERROR: No genes can map to cytoband")
        
    return windows
    
def create_genomic_context_matrix(barcodes, tokenized_sequences, gene_symbol_file, max_length, depth, pad_token_id, seed, custom_tokenizer):
    window_id_list = []
    barcode_list = []
    position_ids_list = []
    all_windows = []
    window_id_counter = 0

    for barcode, seq in zip(barcodes, tokenized_sequences):
        windows = generate_genomic_context(seq, gene_symbol_file, max_length, depth, pad_token_id, seed, custom_tokenizer)
        all_windows.extend(windows)  # Collect all windows to return later
        
        for window in windows:
            if len(window) != max_length:
                raise ValueError(f"Window length mismatch: expected {max_length}, got {len(window)}")
            
            window_id = f"{barcode}_window_{window_id_counter}"
            window_id_counter += 1
            window_id_list.append(window_id)
            barcode_list.append(barcode)
            position_ids_list.append(list(range(len(window))))
    
    if len(window_id_list) != len(all_windows):
        raise ValueError(f"Window ID list length ({len(window_id_list)}) does not match number of windows ({len(all_windows)})")

    window_id_barcode_df = pd.DataFrame({
        'window_id': window_id_list,
        'cell_barcode': barcode_list,
    })
    

    return window_id_barcode_df, all_windows


def prediction_genomic_context_matrix(barcodes, gene_symbol_file, tokenized_sequences, custom_tokenizer, max_length, depth, pad_token_id, y_encoded, rank, targetname, seed=None):
    window_id_list = []
    barcode_list = []
    allwindows = []
    labels_list = []
    gpuid = str(rank)
    position_ids_list = []
    window_id_counter = 0
    window_id_mapping = []
    if y_encoded is None:
        y_encoded = [None] * len(barcodes)
    for barcode, seq, label in zip(barcodes, tokenized_sequences, y_encoded):
        windows = generate_genomic_context(seq, gene_symbol_file, max_length, depth, pad_token_id, seed, custom_tokenizer)
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