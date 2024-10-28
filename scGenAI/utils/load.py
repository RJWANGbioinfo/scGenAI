import os
import torch
import numpy as np
from transformers import AutoTokenizer, LlamaConfig
from transformers import GPT2Tokenizer, GPT2Config
from transformers import BigBirdTokenizer, BigBirdConfig

from sklearn.preprocessing import LabelEncoder
from ..models.llama import CustomLlamaForSequenceClassification
from ..models.gpt import CustomGPT2ForSequenceClassification
from ..models.bigbird import CustomBigBirdForSequenceClassification
from ..models.scgent import CustomscGenTForSequenceClassification, CustomscGenTConfig, scGenTModelInitializer


def loadle(targetname, model_dir, adata_val):
    if targetname != "NOTROUBLESHOOT":
        # le_classes = np.load(os.path.join(model_dir, 'label_encoder_classes.npy'), allow_pickle=True)
        le_classes = loadleclass(model_dir)
        le = LabelEncoder()
        le.classes_ = le_classes
        y_encoded_val = le.transform(adata_val.obs[targetname])  # Encode true labels
    else:
        le = None
        y_encoded_val = None
    return le, y_encoded_val
    
def load_model_state(model, state_dict_path, device, nouse_keys=None):
    state_dict = torch.load(state_dict_path, map_location=device)
    is_ddp_model = all(key.startswith('module.') for key in state_dict.keys())
    if is_ddp_model:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if nouse_keys is not None:
        state_dict = {k: v for k, v in state_dict.items() if k not in nouse_keys}
    model.load_state_dict(state_dict)
    return model
    
def extract_classifier_dimensions(model_dir):
    state_dict = torch.load(os.path.join(model_dir, 'scGenAI_model.pt'), map_location='cpu')
    classifier_weight_key = 'classifier.weight'
    if 'module.classifier.weight' in state_dict:
        classifier_weight_key = 'module.classifier.weight'
    classifier_weight = state_dict[classifier_weight_key]
    num_labels = classifier_weight.size(0)
    hidden_size = classifier_weight.size(1)
    return hidden_size, num_labels
    
def loadgenes(model_dir):
    trained_genes = np.load(os.path.join(model_dir, 'trained_genes.npy'), allow_pickle=True)
    return trained_genes
    
def loadRNAADTgenes(model_dir):
    trainedRNA_genes = np.load(os.path.join(model_dir, 'trainedRNA_genes.npy'), allow_pickle=True)
    trainedADT_genes = np.load(os.path.join(model_dir, 'gene_list_to_emphasize.npy'), allow_pickle=True)
    return trainedRNA_genes, trainedADT_genes
    
def loadtoken(model_dir, modeltype):
    if modeltype=="llama":
        model_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    elif modeltype=="gpt":
        model_tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    elif modeltype=="bigbird":
        model_tokenizer = BigBirdTokenizer.from_pretrained(model_dir)
    elif modeltype=="scgent":
        model_initializer = scGenTModelInitializer()
        model_tokenizer = model_initializer.get_tokenizer()
    return model_tokenizer
    
def loadleclass(model_dir):
    le_classes = np.load(os.path.join(model_dir, 'label_encoder_classes.npy'), allow_pickle=True)

def updateconfig(model_config, custom_tokenizer, targetname, le):
    model_config.vocab_size = custom_tokenizer.vocab_size
    if targetname != "NOTROUBLESHOOT":
        model_config.num_labels = len(le.classes_)
    return model_config

    
def loadmodel(model_dir, modeltype, custom_tokenizer, targetname, le, class_x, class_y, device):
    nouse_keys=None
    if modeltype=="llama":
        model_config = LlamaConfig.from_pretrained(model_dir)
        model_config = updateconfig(model_config, custom_tokenizer, targetname, le)
        model = CustomLlamaForSequenceClassification(model_config,class_x, class_y)
    elif modeltype=="gpt":
        model_config = GPT2Config.from_pretrained(model_dir)
        model_config = updateconfig(model_config, custom_tokenizer, targetname, le)
        model = CustomGPT2ForSequenceClassification(model_config)
    elif modeltype=="bigbird":
        model_config = BigBirdConfig.from_pretrained(model_dir)
        model_config = updateconfig(model_config, custom_tokenizer, targetname, le)
        model = CustomBigBirdForSequenceClassification(model_config)
    elif modeltype=="scgent":
        model_config = CustomscGenTConfig.from_pretrained(model_dir)
        model_config = updateconfig(model_config, custom_tokenizer, targetname, le)
        model = CustomscGenTForSequenceClassification(model_config)
        nouse_keys=['transformer.lm_head.weight']
        
    model = load_model_state(model, os.path.join(model_dir, 'scGenAI_model.pt'), device, nouse_keys=nouse_keys)
    model.transformer.resize_token_embeddings(custom_tokenizer.vocab_size)
    return model_config, model
    
def loadmodelforfinetune(model, custom_tokenizer, model_dir, modeltype, device):
    nouse_keys=None
    if modeltype=="scgent":
        nouse_keys=['transformer.lm_head.weight']
        
    model = load_model_state(model, os.path.join(model_dir, 'scGenAI_model.pt'), device, nouse_keys=nouse_keys)
    model.transformer.resize_token_embeddings(custom_tokenizer.vocab_size)
    return model
    
def loadpretrain(model_dir, modeltype):
    class_x, class_y = extract_classifier_dimensions(model_dir)
    le = LabelEncoder()
    trained_classes = np.load(os.path.join(model_dir, 'label_encoder_classes.npy'), allow_pickle=True)
    trained_genes = np.load(os.path.join(model_dir, 'trained_genes.npy'), allow_pickle=True)
    le.classes_ = trained_classes
    if modeltype=="llama":
        model_config = LlamaConfig.from_pretrained(model_dir)
        model = CustomLlamaForSequenceClassification(model_config, class_x, class_y)
    elif modeltype=="gpt":
        model_config = GPT2Config.from_pretrained(model_dir)
        model = CustomGPT2ForSequenceClassification(model_config, class_x, class_y)
    elif modeltype=="bigbird":
        model_config = BigBirdConfig.from_pretrained(model_dir)
        model = CustomBigBirdForSequenceClassification(model_config, class_x, class_y)
    elif modeltype=="scgent":
        model_config = CustomscGenTConfig.from_pretrained(model_dir)
        model = CustomscGenTForSequenceClassification(model_config, class_x, class_y)
    model_tokenizer = loadtoken(model_dir, modeltype)
    return trained_classes, trained_genes, model_config, model, model_tokenizer, le, class_x, class_y
    