import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW 
import torch.distributed as dist
from transformers import logging as transformers_logging
from sklearn.preprocessing import LabelEncoder
from loguru import logger


from ..models.llama import CustomLlamaForSequenceClassification, LlamaModelInitializer
from ..models.gpt import CustomGPT2ForSequenceClassification, GPTModelInitializer
from ..models.bigbird import CustomBigBirdForSequenceClassification, BigBirdModelInitializer
from ..models.scgent import CustomscGenTForSequenceClassification, scGenTModelInitializer

from ..data.preprocess import Preprocessor, GeneExpressionTokenizer
from ..data.postprocess import Postprocessor, TrainSummary
from ..context import random_context
from ..context import genomic_context
from ..context import biofounction_context
from ..context import emphasize_genes

from ..utils.distributed import setup_distributed, cleanup, savemodel, savelog
from ..config import Config
transformers_logging.set_verbosity_error()

# Helper function to prepare tokenized data for training
def prepare_training_data(adata_train, config, window_id_barcode_df_train, barcodes_train, input_ids_train, y_encoded_train):
    
    # Prepare labels and position IDs using the pre-generated windows
    labels_list_train = []
    position_ids_list_train = []

    for i, label in enumerate(y_encoded_train):
        num_windows = len([wid for wid in window_id_barcode_df_train['cell_barcode'] if wid == barcodes_train[i]])
        labels_list_train.extend([label] * num_windows)
        position_ids_list_train.extend([list(range(config.max_length)) for _ in range(num_windows)])

    input_ids_train = torch.tensor(input_ids_train, dtype=torch.long)
    labels_train = torch.tensor(labels_list_train, dtype=torch.long)
    position_ids_train = torch.tensor(position_ids_list_train, dtype=torch.long)
    
    return input_ids_train, labels_train, position_ids_train

def CreateTensorDataset(custom_tokenizer, train_sequences, adata_train, y_encoded_train, config, Ttype):
    if config.mode=='Train':
        save_dir=config.model_dir
    elif config.mode=='Finetune':
        save_dir=config.finetune_dir
    # Tokenize sequences
    tokenized_train_sequences = custom_tokenizer(train_sequences)

    # Generate barcodes
    barcodes_train = adata_train.obs.index.tolist()

    # Create window-to-barcode DataFrames
    if config.context_method in ["random", "genelist"]:
        window_id_barcode_df_train, input_ids_train = random_context.create_random_context_matrix(barcodes_train, tokenized_train_sequences, config.max_length, stride=config.stride, pad_token_id=custom_tokenizer.pad_token_id)

    elif config.context_method == "genomic":
        window_id_barcode_df_train, input_ids_train = genomic_context.create_genomic_context_matrix(barcodes_train, tokenized_train_sequences, config.cytofile, config.max_length, config.depth,  custom_tokenizer.pad_token_id, config.seed, custom_tokenizer)
    
    elif config.context_method == "biofounction":
        window_id_barcode_df_train, input_ids_train = biofounction_context.create_biofounction_context_matrix(barcodes_train, tokenized_train_sequences, config.gmtfile, config.max_length, config.depth,  custom_tokenizer.pad_token_id, config.seed, custom_tokenizer)

    # Save barcode windows
    window_id_barcode_df_train.to_csv(os.path.join(save_dir, 'context_matrix_'+Ttype+'.csv'), index=False)

    # Prepare input data for training and validation
    input_ids_train, labels_train, position_ids_train = prepare_training_data(adata_train, config, window_id_barcode_df_train, barcodes_train, input_ids_train, y_encoded_train)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(input_ids_train, labels_train, position_ids_train)
    if Ttype == "val":
        shuffle=False
    else:
        shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle)
    return train_dataset, train_loader, window_id_barcode_df_train
    
def DataPreprocessing(config):

    
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

    # Run preprocessing pipeline
    if config.multiomics == "No":
        adata_train, adata_val, train_sequences, val_sequences = preprocessor.preprocess_data()
    else:
        adata_train, adata_val, train_sequences, val_sequences, gene_list_to_emphasize = preprocessor.preprocess_multiomicsdata()
        

    le = LabelEncoder()
    y_encoded_train = le.fit_transform(adata_train.obs[config.target_feature])
    
    if config.model_backbone_name == 'llama':
        model_initializer = LlamaModelInitializer(cache_dir=config.cache_dir)
        
    elif  config.model_backbone_name == 'gpt':
        model_initializer = GPTModelInitializer(cache_dir=config.cache_dir)

    elif  config.model_backbone_name == 'bigbird':
        model_initializer = BigBirdModelInitializer(cache_dir=config.cache_dir)
        
    elif  config.model_backbone_name == 'scgent':
        model_initializer = scGenTModelInitializer(cache_dir=config.cache_dir)

    model_tokenizer = model_initializer.get_tokenizer()        
    custom_tokenizer = GeneExpressionTokenizer(base_tokenizer=model_tokenizer, gene_names=adata_train.var_names, bins=config.num_bins)
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
    return adata_train, adata_val, train_loader, val_loader, custom_tokenizer, model_tokenizer, le, train_dataset

# Training and evaluation function using DDP
def model_train_and_eval(rank, world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer):
    
    if torch.cuda.device_count() > 0 and world_size > 0:
        # Distributed setup
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        # CPU setup
        device = torch.device('cpu')

    best_dir = os.path.join(config.model_dir, "best_model")

    config.num_labels=len(le.classes_)
    
    if config.model_backbone_name == 'llama':
        # LLaMA model configuration
        model_config = LlamaModelInitializer.get_model_config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            vocab_size=custom_tokenizer.vocab_size,
            max_length=config.max_length
        )
 
        model_config.num_labels = len(LabelEncoder().fit(adata_train.obs[config.target_feature]).classes_)
        # Initialize model and optimizer
        model = CustomLlamaForSequenceClassification(model_config).to(device)
        
    elif config.model_backbone_name == 'gpt':
        model_config = GPTModelInitializer.get_model_config(
            n_embd=config.hidden_size,  # Change hidden_size to n_embd
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            vocab_size=custom_tokenizer.vocab_size,
            num_labels=config.num_labels
        )
        model_config.num_labels = len(LabelEncoder().fit(adata_train.obs[config.target_feature]).classes_)
        # Initialize model and optimizer
        model = CustomGPT2ForSequenceClassification(model_config).to(device)

    elif config.model_backbone_name == 'bigbird':
        model_config = BigBirdModelInitializer.get_model_config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_heads,
            num_hidden_layers=config.num_layers,
            vocab_size=custom_tokenizer.vocab_size,
            max_position_embeddings=config.max_length,
            num_labels=config.num_labels
        )
        model_config.num_labels = len(LabelEncoder().fit(adata_train.obs[config.target_feature]).classes_)
        model = CustomBigBirdForSequenceClassification(model_config).to(device)
        
    elif  config.model_backbone_name == 'scgent':
        model_config = scGenTModelInitializer.get_model_config(
            n_embd=config.hidden_size,  # Change hidden_size to n_embd
            n_head=config.num_heads,
            n_layer=config.num_layers,
            vocab_size=custom_tokenizer.vocab_size,
            max_seq_length=config.max_length,
            num_labels=config.num_labels
        )
        model_config.num_labels = len(LabelEncoder().fit(adata_train.obs[config.target_feature]).classes_)
        model = CustomscGenTForSequenceClassification(model_config).to(device)
    

    model.transformer.resize_token_embeddings(model_config.vocab_size)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total number of trainable parameters: {total_params}; Max Sequence Legth: {int(config.max_length/2)} genes')
    
    model = model.to(device)
    if world_size > 0 and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
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
                rank, model, val_loader, device, le.classes_, config.model_dir, epoch, config.num_epochs, train_loss, train_accuracy
            )
            torch.save(results_list, os.path.join(config.model_dir, f"results_df_epoch_{epoch}_rank_{rank}.pt"))
        else:
            window_val_accuracy = train_accuracy
        if (rank == 0) and (window_val_accuracy >= best_window_val_accuracy):
            best_window_val_accuracy = window_val_accuracy
            savemodel(best_dir, model, model_tokenizer, le, custom_tokenizer, adata_train, model_config)
                
        
    if rank == 0:
        last_dir = os.path.join(config.model_dir, "last_model")
        savemodel(last_dir, model, model_tokenizer, le, custom_tokenizer, adata_train, model_config)
        np.savetxt(os.path.join(config.model_dir, 'epoch_losses.txt'), epoch_losses, delimiter='\t', header='epoch\tloss')
        np.savetxt(os.path.join(config.model_dir, 'epoch_accuracies.txt'), epoch_accuracies, delimiter='\t', header='epoch\taccuracy')
    if world_size > 0:
        dist.barrier()
        cleanup()
def evaluate_and_save_predictions(rank, model, data_loader, device, le_classes, model_dir, epoch, Nepochs, train_loss, train_accuracy):
    model.eval()
    total_loss = 0.0
    results_list = []
    correct_predictions = 0
    total_predictions = 0
    window_id_barcode_df = pd.read_csv(os.path.join(model_dir, 'context_matrix_val.csv'))
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f"Evaluating Epoch {epoch+1}/{Nepochs}") as pbar:
            for batch_idx, (inputs, targets, position_ids) in enumerate(data_loader):
                inputs, targets, position_ids = inputs.to(device), targets.to(device), position_ids.to(device)

                outputs = model(input_ids=inputs, labels=targets, position_ids=position_ids)
                loss, logits = outputs if isinstance(outputs, tuple) else (None, outputs)

                if loss is not None:
                    total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                prediction_scores = torch.softmax(logits, dim=-1)

                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)

                for i in range(predictions.size(0)):
                    pos_ids = position_ids[i].tolist()
                    pred = predictions[i].item()
                    score = prediction_scores[i][pred].item()
                    true_label = targets[i].item()

                    window_id = window_id_barcode_df.iloc[batch_idx * predictions.size(0) + i]['window_id']
                    cell_barcode = window_id_barcode_df.iloc[batch_idx * predictions.size(0) + i]['cell_barcode']

                    results_list.append({
                        'cell_barcode': cell_barcode,
                        'context_id': window_id,
                        'position_ids': pos_ids,
                        'prediction': le_classes[pred],
                        'prediction_score': score,
                        'true_label': le_classes[true_label]
                    })

                pbar.update(1)
    
    window_val_accuracy = correct_predictions / total_predictions
    if rank==0:
        logger.info(f'Epoch {epoch+1}/{Nepochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Context-level Val Loss: {total_loss/len(data_loader):.4f}, Context-level Val Accuracy: {window_val_accuracy:.4f}')

    return total_loss / len(data_loader), window_val_accuracy, pd.DataFrame(results_list)


def postprocessing_summary(world_size, config, adata_val):
    # Initialize the Postprocessor
    if config.mode=='Train':
        save_dir=config.model_dir
    elif config.mode=='Finetune':
        save_dir=config.finetune_dir
    postprocessor = Postprocessor(model_dir = save_dir, target_name=config.target_feature, world_size=world_size, keepIntermediateFiles=config.keepIntermediateFiles)
    if config.evaluate_during_training:
        # Summarize the results across all epochs by calculating validation accuracy for each epoch
        postprocessor.summarize_epochs(adata_val, config.num_epochs)
        postprocessor.combine_epoch_results()
        postprocessor.move_intermediate_files()
    else:
        postprocessor.combine_trainonly_results()

def run_training_from_config(config_file):
    
    config = Config(config_file=config_file)
    
    if config.savelog!="No":
        savelog("Train", config)
        
    for key, value in config.__dict__.items():
        if '_dir' in key or 'file' in key:
            if os.path.isfile(value) or os.path.isdir(value):
                print(f"{key}: {value}")
        elif key!='master_addr' and key!='master_port':
            if value!='NOTROUBLESHOOT':
                print(f"{key}: {value}")
    # Preprocess data

    
    adata_train, adata_val, train_loader, val_loader, custom_tokenizer, model_tokenizer, le, train_dataset  = DataPreprocessing(config)

    # Start training
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Run training on multiple GPUs with distributed setup
        torch.multiprocessing.spawn(
            model_train_and_eval,
            args=(world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer),
            nprocs=world_size,
            join=True
        )
    else:
        # Run training on a single GPU or CPU
        model_train_and_eval(0, world_size, config, adata_train, adata_val, train_loader, train_dataset, val_loader, custom_tokenizer, le, model_tokenizer)
    
    # Postprocess results
    postprocessing_summary(world_size, config, adata_val)
    
    summary = TrainSummary(config.model_dir)
    summary.plot(saveplot=True)
