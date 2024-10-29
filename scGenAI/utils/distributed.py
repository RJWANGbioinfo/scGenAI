import os
import torch
import torch.distributed as dist
import numpy as np
from transformers import PreTrainedTokenizer
from loguru import logger

from datetime import datetime
import random
import string
import logging

# Distributed setup functions
def setup_distributed(rank, world_size):
    """
    Setup the environment for Distributed Data Parallel (DDP) training.

    Args:
        rank (int): The rank of the current process (GPU ID).
        world_size (int): The total number of processes (GPUs) participating in the training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Distributed setup complete for GPU {rank}, Total GPUs used: {world_size}")

def cleanup():
    """
    Clean up the Distributed Data Parallel (DDP) environment.
    """
    dist.destroy_process_group()
    logger.info("Cleaned up DDP environment")

# Model saving utility
def savemodel(m_dir, model, m_tokenizer: PreTrainedTokenizer, le, custom_tokenizer, adata_train, config):
    """
    Save the model, tokenizer, and other relevant components to the specified directory.

    Args:
        m_dir (str): Directory to save the model and related components.
        model (torch.nn.Module): Trained PyTorch model to save.
        m_tokenizer (PreTrainedTokenizer): The pre-trained tokenizer used for the model.
        le (LabelEncoder): The label encoder used for encoding target labels.
        custom_tokenizer (GeneExpressionTokenizer): Custom tokenizer for gene expression data.
        adata_train (AnnData): Training data containing gene names.
        config (LlamaConfig): Model configuration used for training.
    """
    os.makedirs(m_dir, exist_ok=True)

    # Save model state
    model_state_path = os.path.join(m_dir, 'scGenAI_model.pt')
    torch.save(model.state_dict(), model_state_path)
    logger.info(f"Model state saved to {model_state_path}")

    # Save tokenizer
    tokenizer_path = m_dir
    m_tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Save custom tokenizer vocabularies
    np.save(os.path.join(m_dir, 'label_encoder_classes.npy'), le.classes_)
    np.save(os.path.join(m_dir, 'gene_vocab.npy'), custom_tokenizer.gene_vocab)
    np.save(os.path.join(m_dir, 'expression_vocab.npy'), custom_tokenizer.expression_vocab)
    np.save(os.path.join(m_dir, 'pad_token_id.npy'), np.array([custom_tokenizer.pad_token_id]))
    np.save(os.path.join(m_dir, 'trained_genes.npy'), adata_train.var_names)
    # np.savez(
        # os.path.join(m_dir, 'vacab_encoder.npz'),
        # label_encoder_classes=le.classes_,
        # gene_vocab=custom_tokenizer.gene_vocab,
        # expression_vocab=custom_tokenizer.expression_vocab,
        # pad_token_id=np.array([custom_tokenizer.pad_token_id]),
        # trained_genes=adata_train.var_names
    # )
    logger.info("Custom vocab and label encoder saved")



    # Save model configuration
    config.save_pretrained(m_dir)
    logger.info(f"Model configuration saved to {m_dir}")

# Barrier synchronization utility
def synchronize():
    """
    Synchronize processes in a distributed setup.
    Ensures all processes have reached the same point before proceeding.
    """
    if dist.is_initialized():
        dist.barrier()
        logger.info("Processes synchronized")

def savelog(Mode, config):
    """
    Save log file with a timestamp and random 5-character string.
    
    Args:
        Mode (str): Mode of operation, such as "Train" or "Predict".
        config (Config): Configuration object with log_dir attribute.
    """
    # Generate timestamp and random 5-character string
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    # Define the log file name with timestamp and random string
    log_file_name = f"{timestamp}-{random_str}-{Mode}.log"

    # Ensure log directory exists
    os.makedirs(config.log_dir, exist_ok=True)

    # Define the log file path
    log_file_path = os.path.join(config.log_dir, log_file_name)

    # Configure loguru to log both to console and file
    logger.add(log_file_path, backtrace=True, diagnose=True)
