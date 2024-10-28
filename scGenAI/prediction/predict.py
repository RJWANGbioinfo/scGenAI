import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import logging as transformers_logging
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from ..data.preprocess import Preprocessor, GeneExpressionTokenizer
from ..data.postprocess import Postprocessor
from ..context import random_context
from ..context import genomic_context
from ..context import biofounction_context
from ..context import emphasize_genes
from ..utils.distributed import setup_distributed, cleanup, savelog
from ..utils.load import *
from ..config import Config, update_predconfig
transformers_logging.set_verbosity_error()

def PredLoader(barcodes_val, tokenized_val_sequences, custom_tokenizer, y_encoded_val, rank, world_size, config):
    if config.context_method in ["random", "genelist"]:
        window_id_barcode_df_val, allwindows, labels_list, window_id_list, position_ids_list, window_id_mapping = random_context.prediction_random_context_matrix(
            barcodes_val, tokenized_val_sequences, config.max_length, custom_tokenizer.pad_token_id, y_encoded_val, rank, config.target_feature
        )
    elif config.context_method == "genomic":
        window_id_barcode_df_val, allwindows, labels_list, window_id_list, position_ids_list, window_id_mapping = genomic_context.prediction_genomic_context_matrix(
            barcodes_val, config.cytofile, tokenized_val_sequences, custom_tokenizer, config.max_length, config.depth, custom_tokenizer.pad_token_id, y_encoded_val, rank, config.target_feature, seed=config.seed)
    elif config.context_method == "biofounction":
        window_id_barcode_df_val, allwindows, labels_list, window_id_list, position_ids_list, window_id_mapping = biofounction_context.prediction_biofounction_context_matrix(
            barcodes_val, config.gmtfile, tokenized_val_sequences, custom_tokenizer, config.max_length, config.depth, custom_tokenizer.pad_token_id, y_encoded_val, rank, config.target_feature, seed=config.seed)

    input_ids_pre = torch.tensor(allwindows, dtype=torch.long)
    position_ids_pre = torch.tensor(position_ids_list, dtype=torch.long)
    window_ids_pre = torch.tensor(window_id_list, dtype=torch.long)
    predict_dataset = TensorDataset(input_ids_pre, position_ids_pre, window_ids_pre)
    sampler = DistributedSampler(predict_dataset, num_replicas=world_size, rank=rank)
    predict_loader = DataLoader(predict_dataset, batch_size=config.batch_size, sampler=sampler)
    return predict_loader, window_id_barcode_df_val, window_id_mapping
    
def PredictionPreprocessing(config):
    if config.model_backbone_name == 'llama':
        class_x, class_y = extract_classifier_dimensions(config.model_dir)
    else:
        class_x = None
        class_y = None
    trained_genes = loadgenes(config.model_dir)
    preprocessor = Preprocessor(
        predict_file=config.predict_file, 
        predict_ADTfile=config.predict_ADTfile, 
        seed=config.seed, 
        max_length=config.max_length,
        bins=config.num_bins,
        trained_genes=trained_genes,
        model_dir=config.model_dir
    )
    config.stride = int(config.max_length / 2)
    
    if config.multiomics == "No":
        adata_pre, pre_sequences = preprocessor.preprocess_prediction()
    else:
        adata_pre, pre_sequences, gene_list_to_emphasize = preprocessor.preprocess_multiomicsprediction()
        
    model_tokenizer = loadtoken(config.model_dir, config.model_backbone_name)
    le, y_encoded_pre = loadle(config.target_feature, config.model_dir, adata_pre)

    custom_tokenizer = GeneExpressionTokenizer(base_tokenizer=model_tokenizer, gene_names=adata_pre.var_names, bins=config.num_bins)
    if config.multiomics != "No":
        pre_sequences = emphasize_genes.emphasize_genes_byfactor(pre_sequences, gene_list_to_emphasize, custom_tokenizer, emphasis_factor=config.depth, seed=config.seed)
    model_tokenizer.pad_token = model_tokenizer.eos_token
    tokenized_pre_sequences = custom_tokenizer(pre_sequences)
    barcodes_pre = adata_pre.obs.index.tolist()
    return adata_pre, pre_sequences, tokenized_pre_sequences, barcodes_pre, custom_tokenizer, y_encoded_pre, class_x, class_y, le

def Prediction(rank, adata_val_aligned, val_sequences, tokenized_val_sequences, barcodes_val, custom_tokenizer, 
    world_size, y_encoded_val, le, class_x, class_y, config):
    setup_distributed(rank, world_size)
    device = torch.device('cuda', rank)
    
    predict_loader, window_id_barcode_df_val, window_id_mapping = PredLoader(barcodes_val, tokenized_val_sequences, \
                                                                             custom_tokenizer, y_encoded_val, rank, world_size, config)
    model_config, model = loadmodel(config.model_dir, config.model_backbone_name, custom_tokenizer, config.target_feature, le, class_x, class_y, device)
    model = model.to(device)
    if world_size > 0 and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()
    results_list = []
    with torch.no_grad():
        with tqdm(total=len(predict_loader), desc=f"GPU {rank} Predicting") as pbar:
            for batch_idx, (inputs, position_ids, window_ids) in enumerate(predict_loader):
                inputs, position_ids, window_ids = inputs.to(device), position_ids.to(device), window_ids.to(device)
                logits = model(input_ids=inputs, position_ids=position_ids)
                predictions = torch.argmax(logits, dim=-1)
                prediction_scores = torch.softmax(logits, dim=-1)
                for i in range(predictions.size(0)):
                    pos_ids = position_ids[i].tolist()
                    pred = predictions[i].item()
                    score = prediction_scores[i][pred].item()
                    # Retrieve both numeric and string-based window IDs
                    numeric_window_id = window_ids[i].item()
                    context_str = next(filter(lambda x: x[0] == numeric_window_id, window_id_mapping))[1]
                    cell_barcode = context_str.split("_", 1)[1].rsplit("_window_", 1)[0]
                    result = {
                        'cell_barcode': cell_barcode,
                        'context_id': context_str,
                        'prediction': np.load(os.path.join(config.model_dir, 'label_encoder_classes.npy'), allow_pickle=True)[pred],
                        'prediction_score': score
                    }
                    if config.target_feature != "NOTROUBLESHOOT":
                        true_label = window_id_barcode_df_val[window_id_barcode_df_val.window_id == numeric_window_id].Label.values[0]
                        true_label = le.inverse_transform([true_label])[0]
                        result['true_label'] = true_label
                    results_list.append(result)
                pbar.update(1)
    results_df = pd.DataFrame(results_list)
    torch.save(results_df, os.path.join(config.model_dir, f"results_df_rank_{rank}.pt"))
    
    if world_size > 0:
        dist.barrier()
        cleanup()

def PostprocessingPrediction(world_size, config, adata_pre):
    postprocessor = Postprocessor(model_dir = config.model_dir, target_name=config.target_feature, world_size=world_size, outputfile=config.outputfile, adata_pre=adata_pre)
    postprocessor.combine_prediction_results()


def run_prediction_from_config(config_file):
    config = update_predconfig(config_file)
    
    if config.savelog!="No":
        savelog("Predict", config)
        
    for key, value in config.__dict__.items():
        if '_dir' in key or 'file' in key:
            if os.path.isfile(value) or os.path.isdir(value):
                print(f"{key}: {value}")
        elif key!='master_addr' and key!='master_port':
            if value!='NOTROUBLESHOOT':
                print(f"{key}: {value}")
    # Preprocess data
    adata_pre, pre_sequences, tokenized_pre_sequences, barcodes_pre, custom_tokenizer, y_encoded_pre, class_x, class_y, le = PredictionPreprocessing(config)

    # Start training
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Run training on multiple GPUs with distributed setup
        torch.multiprocessing.spawn(
            Prediction,
            args=(adata_pre, pre_sequences, tokenized_pre_sequences, barcodes_pre, custom_tokenizer, 
            world_size, y_encoded_pre, le, class_x, class_y, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Run training on a single GPU or CPU
        Prediction(0, adata_pre, pre_sequences, tokenized_pre_sequences, barcodes_pre, custom_tokenizer, world_size, y_encoded_pre, le, class_x, class_y, config)
    
    # Postprocess results
    PostprocessingPrediction(world_size, config, adata_pre)
