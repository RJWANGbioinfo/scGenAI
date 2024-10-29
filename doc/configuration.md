
# Configuration Guide

This document explains how to manage the configuration files for **scGenAI**. You can configure the package by either:
- **Editing the YAML configuration file manually**, using the provided template.
- **Generating the YAML configuration file programmatically** via the `config.py` script using command-line arguments.

## Option 1: Editing the Configuration File from template

The configuration file is a YAML file that defines various settings such as input files, model parameters, and training details. We strongly suggest to start the editing from the config template that prebuilt: [**config_templates**](../examples/config_templates/) 

As a quick start, the user may consider revising the template based on the corresponding training purpose, as shown in the table below:

| Analysis Purpose                               | Configuration Template                                                                                                                  |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Healthy cell type training                     | [`config_Train_genelist_template_llama.yaml`](../examples/config_templates/config_Train_genelist_template_llama.yaml)                    |
| Disease/Cancer cell type training and prediction | [`config_Train_biofounction_context_template_llama.yaml`](../examples/config_templates/config_Train_biofounction_context_template_llama.yaml) or [`config_Train_genomic_context_template_llama.yaml`](../examples/config_templates/config_Train_genomic_context_template_llama.yaml) |
| Prediction                                     | [`config_Prediction_template.yaml`](../examples/config_templates/config_Prediction_template.yaml)                                        |
| Training using multi-omics data                | [`config_Train_MultiOmicsData_template.yaml`](../examples/config_templates/config_Train_MultiOmicsData_template.yaml)                    |
| Fine-tuning using a pretrained model           | [`config_Finetune_MultiOmicsData_template.yaml`](../examples/config_templates/config_Finetune_MultiOmicsData_template.yaml)              |


### Example YAML Configuration (Train Mode)

A example of the format in the config file shown below:

```yaml
# Mode
mode: "Train"

# Directories
cache_dir: "/path_to_your_cache_dir/cache" ## cache dir to save the model template files
model_dir: "/path_to_your_output_model_dir/" ## output model dir
log_dir: "/path_to_your_log_dir/logs/"

#### Input data files ####
train_file: "/path_to_your_trainfile/train.h5ad"
val_file: "/path_to_your_valfile/val.h5ad" ## Optional 

#### General setting ####
savelog: "Yes"
target_feature: "ct"  # Target name for prediction
num_bins: 10  # Bins for gene expression discretization

#### Model template and context method setting ####
model_backbone_name: "llama" ### "llama", "gpt", "bigbird", "scgent"
model_backbone_size: "small" ### "small", "normal", "large". Suggest "small" for llama
max_length: 5120 
context_method: "random"

#### Other settings ####
min_cells: 50  # suggest 50
batch_size: 1 # set this based on GPU memory, higher batch_size higher training speed, but also much more GPU memory will be used
learning_rate: 1e-5  # suggest 1e-5
num_epochs: 30  # suggest 30
```

### Explanation of Parameters

- **`mode`**: Operating mode of the package. It can be `Train`, `Predict`, or `Finetune`. 
- **`train_file`**: The path to the primary training data file in `.h5ad` format. This is required in `Train` and `Finetune` modes.
- **`val_file`**: (Optional) The validation data file used during training. 
- **`batch_size`**: The batch size for training the model.
- **`learning_rate`**: The learning rate used during training.
- **`num_epochs`**: The number of epochs for training the model.
- **`model_backbone_name`**: The backbone model to be used. Choose from `gpt`, `bigbird`, `llama`, or `scGenT`.
- **`model_backbone_size`**: Size of the backbone model (`small`, `normal`, or `large`).
- **`context_method`**: Method for generating context for input sequences. Choices include `random`, `genomic`, `biofounction`, or `genelist`.
- **`target_feature`**: The feature to predict (e.g., `celltype`).
- **`output_dir`**: Directory to save the model outputs.
- **`log_dir`**: Directory to store log files.
- **`cache_dir`**: Directory to cache models during training and prediction.


For more details on input files and parameters, please see the [Input Documentation](input.md).

_________

## Option 2: Generate Configuration through command line using config.py

You can generate the configuration file programmatically using the `config.py` script with command-line arguments. This approach ensures that you can dynamically override default values based on your specific requirements.

### Command-Line Argument Example

Run the following command to generate a YAML configuration file:

```bash
python config.py --mode Train --train_file /rootdir/examples/data/train_data.h5ad \
                 --val_file /rootdir/examples/data/val_data.h5ad --batch_size 2 \
                 --learning_rate 1e-5 --num_epochs 30 --model_backbone_name llama \
                 --model_backbone_size small --context_method genomic \
                 --cytofile /rootdir/examples/data/cytoband_data.txt --target_feature celltype \
                 --outputconfig /rootdir/train_config.yaml
```

This will generate a `train_config.yaml` file based on the provided parameters.

### Available Command-Line Arguments

- `--mode`: Specifies the mode of operation (`Train`, `Predict`, `Finetune`).
- `--train_file`: Path to the training file.
- `--val_file`: Path to the validation file.
- `--evaluate_during_training`: whether to use validation file in training or prediction (`true`, `false`). Default is `true`. Set this to `false` if there is no validation file.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for training.
- `--num_epochs`: Number of training epochs.
- `--model_backbone_name`: Backbone model name (`gpt`, `bigbird`, `llama`, `scGenT`).
- `--model_backbone_size`: Size of the model (`small`, `normal`, `large`).
- `--context_method`: Context generation method (`random`, `genomic`, `biofounction`, `genelist`).
- `--cytofile`: Path to the cytoband data file (if using `genomic` context).
- `--target_feature`: Feature to predict (e.g., `celltype`).
- `--output_dir`: Directory to save outputs.
- `--log_dir`: Directory to save logs.
- `--outputconfig`: The path where the generated configuration YAML file will be saved.


For more details on input files and parameters, please see the [Input Documentation](input.md).

_________

##  Next Step: Run scGenAI

Once the config file is built, the user can now run the scGenAI. Please read the doc of [Run scGenAI](usage.md) for details
