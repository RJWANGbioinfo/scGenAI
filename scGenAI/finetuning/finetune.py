import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW 
import torch.distributed as dist
from transformers import logging as transformers_logging
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from ..training.train import prepare_training_data, CreateTensorDataset, evaluate_and_save_predictions, postprocessing_summary
from ..data.preprocess import Preprocessor, GeneExpressionTokenizer
from ..data.postprocess import Postprocessor, TrainSummary
from ..utils.distributed import setup_distributed, cleanup, savemodel, savelog
from ..utils.load import *
from ..config import Config, update_predconfig
transformers_logging.set_verbosity_error()


def FinetunePreprocessing(config):

    trained_classes, trained_genes, model_config, model, model_tokenizer, le, class_x, class_y = loadpretrain(config.model_dir, config.model_backbone_name)

    preprocessor = Preprocessor(
        train_file=config.train_file, 
        val_file=config.val_file, 
        train_ADTfile=config.train_ADTfile, 
        val_ADTfile=config.val_ADTfile,
        min_cells=config.min_cells, 
        seed=config.seed, 
        max_length=config.max_length,
        subset=config.context_method,
        glstfile=config.glstfile,
        bins=config.num_bins,
        model_dir=config.model_dir
    )

    config.stride = int(config.max_length / 2)
    input_genes, input_classes = preprocessor.extractgene_lable(config.target_feature)
    
    if config.multiomics != "No":
        trained_gene_in_RNA, trained_gene_list_to_emphasize = loadRNAADTgenes(config.model_dir)
        input_gene_in_RNA, input_gene_list_to_emphasize, input_classes = preprocessor.extractgene_lable(config.target_feature)
        input_genes = input_gene_in_RNA + input_gene_list_to_emphasize
        
        newRNAs = [gene for gene in input_gene_in_RNA if gene not in trained_gene_in_RNA]
        newADTs = [gene for gene in input_gene_list_to_emphasize if gene not in trained_gene_list_to_emphasize]

        if len(newRNAs)>0:
            allRNAs = list(trained_gene_in_RNA) + newRNAs
        else:
            allRNAs = list(trained_gene_in_RNA)
            
        if len(newADTs)>0:
            allADTs = list(trained_gene_list_to_emphasize) + newADTs
        else:
            allADTs = list(trained_gene_list_to_emphasize)
        
    newlables = [label for label in input_classes if label not in trained_classes]
    newgenes = [gene for gene in input_genes if gene not in trained_genes]
    
    
    
    if len(newgenes)>0:
        logger.info(f"New genes found: {len(newgenes)}. Updating custom_tokenizer.")
        allgenes = list(trained_genes) + newgenes
    else:
        allgenes = list(trained_genes)


        
    custom_tokenizer = GeneExpressionTokenizer(base_tokenizer=model_tokenizer, gene_names=allgenes, bins=config.num_bins)
    model_config.vocab_size = custom_tokenizer.vocab_size
    
    if len(newlables)>0:
        logger.info(f"New target labels found: {len(newlables)}. Updating label encoder.")
        all_labels = list(le.classes_) + newlables
        le.classes_ = np.array(all_labels)
        model_config.num_labels = len(all_labels)
    else:
        all_labels = list(trained_classes)
        

    # Run preprocessing pipeline
    if config.multiomics == "No":
        adata_train, adata_val, train_sequences, val_sequences = preprocessor.preprocess_data(finetune_genes = allgenes)
    else:
        adata_train, adata_val, train_sequences, val_sequences, gene_list_to_emphasize = preprocessor.preprocess_multiomicsdata(finetune_genes = allRNAs, finetune_adts = allADTs)

    y_encoded_train = le.transform(adata_train.obs[config.target_feature])
    model_tokenizer.pad_token = model_tokenizer.eos_token
    
    # print(custom_tokenizer.vocab_size)
    if config.multiomics != "No":
        train_sequences = emphasize_genes.emphasize_genes_byfactor(train_sequences, gene_list_to_emphasize, custom_tokenizer, emphasis_factor=config.depth, seed=config.seed)
        if val_sequences is not None:
            val_sequences = emphasize_genes.emphasize_genes_byfactor(val_sequences, gene_list_to_emphasize, custom_tokenizer, emphasis_factor=config.depth, seed=config.seed)
    train_dataset, train_loader, window_id_barcode_df_train = CreateTensorDataset(custom_tokenizer, train_sequences, adata_train, y_encoded_train, config, "train")
    
    if config.evaluate_during_training:
        y_encoded_val = le.transform(adata_val.obs[config.target_feature])  
        val_dataset, val_loader, window_id_barcode_df_val = CreateTensorDataset(custom_tokenizer, val_sequences, adata_val, y_encoded_val, config, "val")
    else:
        val_loader = None
    return adata_train, adata_val, train_loader, val_loader, custom_tokenizer, model_tokenizer, le, train_dataset, model, model_config, newlables, newgenes


# Finetune and evaluation function using DDP
def model_finetune_and_eval(rank, world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer, model, model_config, newgenes, newlables):
    if torch.cuda.device_count() > 0 and world_size > 0:
        # Distributed setup
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        # CPU setup
        device = torch.device('cpu')

    best_dir = os.path.join(config.finetune_dir, "best_model")
    model = loadmodelforfinetune(model, custom_tokenizer, config.model_dir, config.context_method, device)
    
    ## update model
    if len(newgenes)>0:
        model.transformer.resize_token_embeddings(custom_tokenizer.vocab_size)
    if len(newlables)>0:
        model.classifier = nn.Linear(custom_tokenizer.vocab_size, model_config.num_labels).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total number of trainable parameters: {total_params}; Max Sequence Legth: {int(config.max_length/2)} genes')
    
    model = model.to(device)
    if world_size > 0 and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    best_window_val_accuracy = 0
    epoch_losses, epoch_accuracies = [], []

    for epoch in range(config.num_epochs):
        model.train()

        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in train_loader:
                batch_input_ids, batch_labels, batch_position_ids = [x.to(device) for x in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=batch_input_ids, labels=batch_labels, position_ids=batch_position_ids)
                loss, logits = outputs if isinstance(outputs, tuple) else (None, outputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

                pbar.set_postfix(
                    GPU_ID=rank,
                    train_loss=epoch_loss/len(train_loader), 
                    train_accuracy=correct_predictions/total_predictions
                )
                pbar.update(1)
        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_accuracy)
        
        if config.evaluate_during_training:
            val_loss, window_val_accuracy, results_list = evaluate_and_save_predictions(
                rank, model, val_loader, device, le.classes_, config.finetune_dir, epoch, config.num_epochs, train_loss, train_accuracy
            )
            torch.save(results_list, os.path.join(config.finetune_dir, f"results_df_epoch_{epoch}_rank_{rank}.pt"))
        else:
            window_val_accuracy = train_accuracy
        if (rank == 0) and (window_val_accuracy >= best_window_val_accuracy):
            best_window_val_accuracy = window_val_accuracy
            savemodel(best_dir, model, model_tokenizer, le, custom_tokenizer, adata_train, model_config)
                
        
    if rank == 0:
        last_dir = os.path.join(config.finetune_dir, "last_model")
        savemodel(last_dir, model, model_tokenizer, le, custom_tokenizer, adata_train, model_config)
        np.savetxt(os.path.join(config.finetune_dir, 'epoch_losses.txt'), epoch_losses, delimiter='\t', header='epoch\tloss')
        np.savetxt(os.path.join(config.finetune_dir, 'epoch_accuracies.txt'), epoch_accuracies, delimiter='\t', header='epoch\taccuracy')
    if world_size > 0:
        dist.barrier()
        cleanup()

def run_finetune_from_config(config_file):
    
    config = update_predconfig(config_file)
    if config.savelog!="No":
        savelog("Finetune", config)
    
    for key, value in config.__dict__.items():
        if '_dir' in key or 'file' in key:
            if os.path.isfile(value) or os.path.isdir(value):
                print(f"{key}: {value}")
        elif key!='master_addr' and key!='master_port':
            if value!='NOTROUBLESHOOT':
                print(f"{key}: {value}")
                
                
    # Preprocess data
    adata_train, adata_val, train_loader, val_loader, custom_tokenizer, model_tokenizer, le, train_dataset, model, model_config, newlables, newgenes  = FinetunePreprocessing(config)

    # Start training
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Run training on multiple GPUs with distributed setup
        torch.multiprocessing.spawn(
            model_finetune_and_eval,
            args=(world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer, model, model_config, newgenes, newlables),
            nprocs=world_size,
            join=True
        )
    else:
        # Run training on a single GPU or CPU
        model_finetune_and_eval(0, world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer, model, model_config, newgenes, newlables)
    
    # Postprocess results
    postprocessing_summary(world_size, config, adata_val)
    
    summary = TrainSummary(config.finetune_dir)
    summary.plot(saveplot=True)
