import numpy as np
import pandas as pd
import random
from ..utils.geneexpparser import *

def slide_context(seq, max_length, pad_token_id):
    windows = []
    stride = int(max_length/2)
    # If the sequence length is less than or equal to max_length, return a single padded window
    if len(seq) <= max_length:
        window = seq + [pad_token_id] * (max_length - len(seq))  # Pad the sequence to the max_length
        windows.append(window)
    else:
        # Otherwise, create multiple windows using the stride
        for i in range(0, len(seq), stride):
            window = seq[i:i + max_length]
            if len(window) < max_length:
                window += [pad_token_id] * (max_length - len(window))  # Pad the last window if it's shorter
            windows.append(window)
            # Stop if the window is shorter than max_length (end of sequence)
            if len(window) < max_length:
                break
    window_ids = [f"window_{i}" for i in range(len(windows))]
    return windows, window_ids

# Random sampling of gene sets, add gene set ID
def generate_biofounction_context(gmt_dict, paired_seq_dict, expressed_genes, max_length=1024, depth=2, pad_token_id=-1, seed=42, gene_vocab=None):
    np.random.seed(seed)

    
    filtered_gmt_dict = {gene_set: [gene for gene in genes if gene in expressed_genes] 
                         for gene_set, genes in gmt_dict.items() if any(gene in expressed_genes for gene in genes)}
    
    filtered_gmt_dict = {gene_set: genes for gene_set, genes in filtered_gmt_dict.items() if genes}

    gene_set_counts = {gene_set: 0 for gene_set in filtered_gmt_dict.keys()}
    windows = []
    window_ids = []
    max_pair = max_length//2
    while any(count < depth for count in gene_set_counts.values()):
        sampled_genes = []
        sampled_gene_sets = set()  # Use a set to ensure unique gene set names

        while len(sampled_genes) < max_pair:
            available_gene_sets = [gene_set for gene_set, count in gene_set_counts.items() if count < depth]
            if not available_gene_sets:
                break

            gene_set = np.random.choice(available_gene_sets)
            genes = filtered_gmt_dict[gene_set]
            
            if len(sampled_genes) + len(genes) <= max_pair:
                sampled_genes.extend(genes)
                sampled_gene_sets.add(gene_set)  # Add the gene set name to the set
                gene_set_counts[gene_set] += 1
            else:
                break
        sampled_genes_expression = [(gene_vocab.get_token_id(gene), paired_seq_dict[gene_vocab.get_token_id(gene)]) for gene in sampled_genes]
        sampled_genes_expression_flat = [token for pair in sampled_genes_expression for token in pair]
        if len(sampled_genes_expression_flat) < max_length:
            sampled_genes_expression_flat.extend([pad_token_id] * (max_length - len(sampled_genes_expression_flat)))

        windows.append(sampled_genes_expression_flat)
        window_ids.append('_'.join(sorted(sampled_gene_sets)))  
        gene_set_counts = {gene_set: count for gene_set, count in gene_set_counts.items() if count < depth}
    return windows, window_ids    

def create_biofounction_context_matrix(barcodes, tokenized_sequences, gmt_file_path, max_length, depth, pad_token_id, seed, custom_tokenizer):
    window_id_list = []
    barcode_list = []
    all_windows = []
    gene_vocab = BiDirectionalDict(custom_tokenizer.gene_vocab)
    gmt_dict = read_gmt_to_dict(gmt_file_path)
    total_cells = len(barcodes)
    # print(f"Total cells to process: {total_cells}")
    for index, (barcode, seq) in enumerate(zip(barcodes, tokenized_sequences), start=1):
        paired_seq_dict = {seq[i]: seq[i+1] for i in range(0, len(seq), 2)}
        
        # Convert tokens to gene names
        genes = [gene_vocab.get_gene_name(token_id) for token_id in seq[::2] if gene_vocab.get_gene_name(token_id) is not None]
        matching_genes = [gene for gene in genes if any(gene in gene_list for gene_list in gmt_dict.values())]
        unmatching_genes = [gene for gene in genes if gene not in matching_genes]
        if matching_genes:
            
            gmt_windows, gmt_window_ids = generate_biofounction_context(gmt_dict, paired_seq_dict, matching_genes, max_length, depth, pad_token_id, seed, gene_vocab)
            window_id_list.extend([f"{barcode}_{'_'.join(set(window_id.split('_')))}" for window_id in gmt_window_ids])
            barcode_list.extend([barcode] * len(gmt_window_ids))
            all_windows.extend(gmt_windows)
            
        else:
            print("Warning: Cell::", barcode, " :No expressed_genes are found in gene set file")
            non_gmt_windows, non_gmt_window_ids = slide_context(seq, max_length, depth, pad_token_id, seed)
            window_id_list.extend([f"{barcode}_{window_id}" for window_id in non_gmt_window_ids])
            barcode_list.extend([barcode] * len(non_gmt_window_ids))
            all_windows.extend(non_gmt_windows)
            
        if len(matching_genes)>0 and len(unmatching_genes)>0:
            
            subset_seq = []
            
            for gene in unmatching_genes:
                # Get the gene token ID from reverse_gene_vocab (using gene name as the key)
                gene_token_id = gene_vocab.get_token_id(gene)
                # Extract the gene token and corresponding expression token from paired_seq_dict
                if gene_token_id in paired_seq_dict:
                    subset_seq.append(gene_token_id)          # Gene token
                    subset_seq.append(paired_seq_dict[gene_token_id])  # Expression token
            non_gmt_windows, non_gmt_window_ids = slide_context(subset_seq, max_length, pad_token_id)
            # Continue processing with the non_gmt_windows and non_gmt_window_ids
            window_id_list.extend([f"{barcode}_{window_id}" for window_id in non_gmt_window_ids])
            barcode_list.extend([barcode] * len(non_gmt_window_ids))
            all_windows.extend(non_gmt_windows)
            
        if index % 100 == 0 or index == total_cells:
            print(f"Processed {index}/{total_cells} cells", end='\r')

    ### Handle the conversion properly, only converting numerical tokens
    def safe_int_conversion(token):
        try:
            return int(token)
        except ValueError:
            return pad_token_id

    all_windows = [[safe_int_conversion(token) for token in window] for window in all_windows]

    window_id_barcode_df = pd.DataFrame({
        'window_id': window_id_list,
        'cell_barcode': barcode_list,
    })

    return window_id_barcode_df, all_windows

def prediction_biofounction_context_matrix(barcodes, gmt_file_path, tokenized_sequences, custom_tokenizer, max_length, depth, pad_token_id, y_encoded, rank, targetname, seed=None):
    labels_list = []
    gpuid = str(rank)
    position_ids_list = []
    window_id_mapping = []
    window_id_list = []
    barcode_list = []
    all_windows = []
    gene_vocab = BiDirectionalDict(custom_tokenizer.gene_vocab)
    gmt_dict = read_gmt_to_dict(gmt_file_path)
    total_cells = len(barcodes)
    if y_encoded is None:
        y_encoded = [None] * len(barcodes)
    # print(f"Total cells to process: {total_cells}")

    for index, (barcode, seq, label) in enumerate(zip(barcodes, tokenized_sequences, y_encoded), start=1):
        cellwlist=[]
        paired_seq_dict = {seq[i]: seq[i+1] for i in range(0, len(seq), 2)}
        
        # Convert tokens to gene names
        genes = [gene_vocab.get_gene_name(token_id) for token_id in seq[::2] if gene_vocab.get_gene_name(token_id) is not None]
        matching_genes = [gene for gene in genes if any(gene in gene_list for gene_list in gmt_dict.values())]
        unmatching_genes = [gene for gene in genes if gene not in matching_genes]
        if matching_genes:
            
            gmt_windows, gmt_window_ids = generate_biofounction_context(gmt_dict, paired_seq_dict, matching_genes, max_length, depth, pad_token_id, seed, gene_vocab)
            window_id_list.extend([f"{gpuid}_{barcode}_window_{'_'.join(set(window_id.split('_')))}" for window_id in gmt_window_ids])
            barcode_list.extend([barcode] * len(gmt_window_ids))
            all_windows.extend(gmt_windows)
            cellwlist.extend(gmt_windows)
            if targetname != "NOTROUBLESHOOT":
                labels_list.extend([label] * len(gmt_windows))  # Only add labels if not in "NOTROUBLESHOOT" mode
        else:
            print("Warning: Cell::", barcode, " :No expressed_genes are found in gene set file")
            non_gmt_windows, non_gmt_window_ids = slide_context(seq, max_length, depth, pad_token_id, seed)
            window_id_list.extend([f"{gpuid}_{barcode}_{window_id}" for window_id in non_gmt_window_ids])
            barcode_list.extend([barcode] * len(non_gmt_window_ids))
            all_windows.extend(non_gmt_windows)
            cellwlist.extend(non_gmt_windows)
            if targetname != "NOTROUBLESHOOT":
                labels_list.extend([label] * len(non_gmt_windows))  # Only add labels if not in "NOTROUBLESHOOT" mode
            # position_ids_list.append(list(range(len(non_gmt_windows))))
        if len(matching_genes)>0 and len(unmatching_genes)>0:
            
            subset_seq = []
            
            for gene in unmatching_genes:
                # Get the gene token ID from reverse_gene_vocab (using gene name as the key)
                gene_token_id = gene_vocab.get_token_id(gene)
                # Extract the gene token and corresponding expression token from paired_seq_dict
                if gene_token_id in paired_seq_dict:
                    subset_seq.append(gene_token_id)          # Gene token
                    subset_seq.append(paired_seq_dict[gene_token_id])  # Expression token
            non_gmt_windows, non_gmt_window_ids = slide_context(subset_seq, max_length, pad_token_id)
            # Continue processing with the non_gmt_windows and non_gmt_window_ids
            window_id_list.extend([f"{gpuid}_{barcode}_{window_id}" for window_id in non_gmt_window_ids])
            barcode_list.extend([barcode] * len(non_gmt_window_ids))
            all_windows.extend(non_gmt_windows)
            cellwlist.extend(non_gmt_windows)
            if targetname != "NOTROUBLESHOOT":
                labels_list.extend([label] * len(non_gmt_windows))  # Only add labels if not in "NOTROUBLESHOOT" mode
        for cw in cellwlist:
            position_ids_list.append(list(range(len(cw))))
        if index % 100 == 0 or index == total_cells:
            print(f"Processed {index}/{total_cells} cells", end='\r')

    ### Handle the conversion properly, only converting numerical tokens
    def safe_int_conversion(token):
        try:
            return int(token)
        except ValueError:
            return pad_token_id

    all_windows = [[safe_int_conversion(token) for token in window] for window in all_windows]

    window_id_barcode_df = pd.DataFrame({
        'window_str_id': window_id_list,
        'cell_barcode': barcode_list,
    })
    if targetname != "NOTROUBLESHOOT":
        window_id_barcode_df['Label'] = labels_list
    window_id_barcode_df=window_id_barcode_df.assign(window_id=window_id_barcode_df.index.tolist())
    for window_id_counter, window_id_str in zip(window_id_barcode_df.index.tolist(),window_id_list):
        window_id_mapping.append((window_id_counter, window_id_str))
    return window_id_barcode_df, all_windows, labels_list, window_id_barcode_df.index.tolist(), position_ids_list, window_id_mapping