
# scGenAI


##### Author: Ruijia Wang
##### Date: 2024-10-14
##### Version: 0.1.0

<p align="center">
  <img src="img/logoedge.png" alt="scGenAI Logo" />
</p>

## About scGenAI
_________

**scGenAI** is a Python package for single-cell RNA sequencing (scRNA-seq) data prediction and analysis using large language models (LLMs). The package allows users to train, fine-tune, and make predictions on single-cell data using transformer-based models, including custom versions of LLaMA, GPT, BigBird, and scGenT. It provides multi-GPU support with PyTorch DistributedDataParallel (DDP).


## Table of Contents
_________

- [I. Installation](#i-installation)
- [II. Quick Start/Tutorials](#ii-quick-start-tutorials)
  - [Option 1. Use scGenAI through notebook or python script](#option-1-use-scgenai-through-notebook-or-python-script)
  - [Option 2. Use scGenAI through Command Line Interface](#option-2-use-scgenai-through-command-line-interface)
- [III. Documentation](#iii-documentation)
- [VI. License](#vi-license)


## I. Installation

_________

To install the package, use the following steps:

0. **Optional: Create a Env for scGenAI:**

    ```bash
    conda create -n scGenAI python==3.10
    conda activate scGenAI
    ```
	
1. **Clone the repository:**

    ```bash
    git clone https://bitbucket.org/vor-compbio/scgenai.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd scgenai
    ```
	

	
3. **Install dependencies, then install scGenAI:**

    ```bash
    pip install -r requirements.txt
    pip install .
    ```


## II. Quick Start/Tutorials

_________

Once installed, `scGenAI` can be accessed through either (1) python IDE or notebook or (2) the command line interface (CLI). You can train, predict, or fine-tune a model by calling the scGenAI in python or CLI commands along with a configuration YAML file containing your settings.

### Option 1. Use scGenAI through notebook or python script

As a quick start, we highly recommend users to begin with the following [tutorials](./tutorials/) and [config template files](./examples/config_templates/) according to the training/prediction purposes:

| Project                          | Tutorial                                 | Training/Finetune Config Template                                                                                                     | Prediction Config Template                                                                 |
|-----------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Modeling healthy cell type        | [TrainData_genelist](./tutorials/TrainData_genelist.ipynb)                                 | [config_Train_genelist](./examples/config_templates/config_Train_genelist_template_llama.yaml)                   | [config_Prediction](./examples/config_templates/config_Prediction_template.yaml) |
| Modeling disease/cancer cell type | [TrainData_biofounction_context](./tutorials/TrainData_biofounction_context.ipynb)         | [config_Train_biofounction_context](./examples/config_templates/config_Train_biofounction_context_template_llama.yaml) or [config_Train_random_context](./examples/config_templates/config_Train_random_context_template_llama.yaml) | [config_Prediction](./examples/config_templates/config_Prediction_template.yaml) |
| Modeling cell genotype            | [TrainData_genomic_context](./tutorials/TrainData_genomic_context.ipynb)                   | [config_Train_genomic_context](./examples/config_templates/config_Train_genomic_context_template_llama.yaml)     | [config_Prediction](./examples/config_templates/config_Prediction_template.yaml) |
| Modeling using multiomics data    | [TrainData_MultiOmics](./tutorials/TrainData_MultiOmics.ipynb)                             | [config_Train_MultiOmicsData](./examples/config_templates/config_Train_MultiOmicsData_template.yaml)                   | [config_prediction_multiOmic](./examples/config_templates/config_prediction_multiOmic_template.yaml) |
| Fine-tune using pretrained model   | [FinetuneData_random_context](./tutorials/FinetuneData_random_context.ipynb)                | [config_Finetune_MultiOmicsData](./examples/config_templates/config_Finetune_MultiOmicsData_template.yaml)             | [config_Prediction](./examples/config_templates/config_Prediction_template.yaml) |



### Option 2. Use scGenAI through Command Line Interface

The CLI supports the following commands:

- **Train a model**:
    ```bash
    scgenai train --config_file <path_to_config.yaml>
    ```

- **Make predictions**:
    ```bash
    scgenai predict --config_file <path_to_config.yaml>
    ```

- **Fine-tune a pre-trained model**:
    ```bash
    scgenai finetune --config_file <path_to_config.yaml>
    ```

## III. Documentation

_________

Please see the full [documentation](./doc/index.md) for the details usage of **scGenAI**.

- [Input Files and parameters](input.md)
- [Configuration](./doc/configuration.md)
- [Run scGenAI](./doc/usage.md)
- [Output Files](./doc/output.md)

## V. License

_________


**scGenAI** is licensed under a custom license for **non-commercial use only**. It is available for individuals, universities, non-profit organizations, educational and government bodies for non-commercial research or journalistic purposes.
