# Input Files and Parameters

This document describes the input files and parameters required by **scGenAI** for different modes of operation: `Train`, `Predict`, and `Finetune`.


_________


## Input Mode

- **`mode`**: Specifies the mode of operation. Choices: `Train`, `Predict`, `Finetune`. Default: `Train`.


_________


## Input Files and Dir

### Input Files

These files are required in most configurations. Depending on the mode (Train, Predict, Finetune), some may be required:

- **`train_file`**: ***Required*** when `mode` is `Train` or `Finetune`. The primary training data file in `.h5ad` format. 
- **`val_file`**: Optional. The validation data file used during training. 
- **`train_ADTfile`**: ***Required*** when `multiomics` is set to `Yes` and  `mode` is `Train` or `Finetune`. The training file for ADT (antibody-derived tags) data. 
- **`val_ADTfile`**: Optional. The validation file for ADT data. 
- **`cytofile`**: ***Required*** when `context_method` is set to `genomic` and  `mode` is `Train`. The cytoband data file used in genomic context. It should be a tab separated files contains the two columns, `gene_symbol` and `cytobandID`.
- **`gmtfile`**: ***Required*** when `context_method` is set to `biofounction` and  `mode` is `Train`. The gmt format file for biofunction context. 
- **`glstfile`**: ***Required*** when `context_method` is set to `genelist` and  `mode` is `Train` or `Finetune`. A gene list file used in genelist context.  It should be a single column no header with gene names.
- **`predict_file`**: ***Required*** when `mode` is `Predict`.  The input file for prediction tasks. 
- **`predict_ADTfile`**: ***Required*** when `multiomics` is set to `Yes` and  `mode` is `Predict`. The ADT file for prediction tasks (if using multi-omics data). 
- **`outputfile`**: ***Required*** when `mode` is `Predict`. The output CSV file where the prediction results will be saved. 
- **`cache_dir`**: ***Required***. Directory to cache models template during training and prediction, when the first time of using model template, the cache files will be saved in this folder, and will be loaded directly for reuse.

### Input Dir

- **`log_dir`**: ***Required*** when `savelog` is set to `Yes`. Directory to store logs. Default: `examples/logs`.
- **`model_dir`**: ***Required*** when `mode` is `Train` or `Finetune`. Directory where models are saved during training, which is also the input folder in the `Finetune` mode.
- **`finetune_dir`**: ***Required*** when `mode` is `Finetune`. Directory used for saving fine-tuned models. Default: `examples/finetune`.

_________


## Input Parameters

### Preprocessing Settings

- **`min_cells`**: Optional. Minimum number of cells required for filtering. Default: `50`.
- **`target_feature`**: ***Required***. The feature to predict (e.g., `celltype`). Default: `celltype`.
- **`multiomics`**: Optional. Whether multi-omics data (e.g., RNA and ADT) is used. Choices: `Yes`, `No`. Default: `No`.
- **`savelog`**: Optional. Whether to save logs to the `log_dir`. Choices: `Yes`, `No`. Default: `No`.
- **`num_bins`**: Number of bins used for gene expression normalization. Default: `10`.

### Model Settings

- **`model_backbone_name`**: ***Required*** when `mode` is `Train`. The backbone model template to use. Choices: `gpt`, `bigbird`, `llama`, `scGenT`. Default: `llama`.
- **`model_backbone_size`**: ***Required*** when `mode` is `Train`. The size of the model. Choices: `small`, `normal`, `large`. Default: `normal`. Suggested size: `small` for `llama` and `bigbird`;  `normal` for `gpt` and `scGenT`.
- **`context_method`**: Method for generating context. Choices: `random`, `genomic`, `biofounction`, `genelist`. Default: `random`.

### Training Settings

- **`batch_size`**: ***Required***. The batch size for training. Default: `8`. We suggest the user begin with 1 to avoid the error of out of GPU memory.
- **`learning_rate`**: Optional. The learning rate for the model trainning. Default: `1e-5`.
- **`num_epochs`**: Optional. Number of training epochs. Default: `30`. We suggest use `30` for training, `20`~`30` for finetune mode.
- **`weight_decay`**: Optional. Weight decay rate for the optimizer. Default: `0.01`.
- **`depth`**: Optional.  Depth for the genomic/biofounction/genelist context method. Default: `2`.
- **`seed`**: Optional. Random seed for reproducibility. Default: `1314521`.
- **`max_length`**: ***Required*** when `mode` is `Train`. Maximum sequence length for tokenization. Default: `1024`. Suggested size: `5120` for `llama`;  `1024` for `gpt`; `4096` for `bigbird`; `2048` for `scGenT`.

_________


##  Next Step: Build the YAML configuration file

All the required parameters must be defined in the YAML configuration file. Please read the doc of [Configuration](configuration.md) for details

