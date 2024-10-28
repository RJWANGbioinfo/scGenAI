
# Run scGenAI

This document explains how to run **scGenAI** for training, fine-tuning, and prediction tasks. You can use **scGenAI** in two ways:
1. **Running through Python scripts**: Leverage the provided tutorials in the `tutorials/` folder.
2. **Running through the command line**: Execute tasks directly via command-line commands.

## Option 1: Running scGenAI through Python

You can run **scGenAI** through the provided tutorial notebooks or by writing your own Python script. The tutorials cover common workflows, such as training, fine-tuning, and prediction.


### Tutorials Jupyter Notebook

As a quick start, we strongly suggest to begin with the following Jupyter notebooks in the `tutorials/` folder guide you through various workflows:

- **Training and Prediction with Biofunction Context Method**: [`TrainData_biofounction_context.ipynb`](../tutorials/TrainData_biofounction_context.ipynb)
- **Training and Prediction with Genomic Context Method**: [`TrainData_genomic_context.ipynb`](../tutorials/TrainData_genomic_context.ipynb)
- **Training and Prediction with Random Context Method**: [`TrainData_random_context.ipynb`](../tutorials/TrainData_random_context.ipynb)
- **Training and Prediction using Multi-omics Data**: [`TrainData_MultiOmics.ipynb`](../tutorials/TrainData_MultiOmics.ipynb)
- **Fine-tuning a pre-trained model**: [`FinetuneData_random_context.ipynb`](../tutorials/FinetuneData_random_context.ipynb)

To explore these, simply open the Jupyter notebook of your choice and follow the steps to run the training, prediction, or fine-tuning process.


### Python Examples 
Here are a few examples of how to run **scGenAI** using Python:

#### Training

```python
from scGenAI.training.train import run_training_from_config

# Path to your configuration file
config_path = "/packagdir/config_templates/config_Train_genelist_template_llama.yaml"

# Run training
run_training_from_config(config_path)
```

#### Prediction

```python
from scGenAI.prediction.predict import run_prediction_from_config

# Path to your configuration file
config_path = "/packagdir/config_templates/config_random_context_template_llama.yaml"

# Run training
run_prediction_from_config(config_path)
```

#### Fine-tune

```python
from scGenAI.finetuning.finetune import run_finetune_from_config

# Path to your configuration file
config_path = "/packagdir/config_templates/config_Finetune_random_context_template_llama.yaml"

# Run training
run_finetune_from_config(config_path)
```

#### Login Huggingface (ONLY required for the FIRST TIME use of llama model template)

Logging into **Hugging Face** with a token is required when using the LLaMA model for the first time. This requirement is specific to LLaMA and does not apply to other model architectures supported by **scGenAI**.

```python
from huggingface_hub import login
import getpass

def login_to_huggingface():
    """
    Prompts the user to securely enter their Hugging Face token and logs them into the Hugging Face Hub.
    """
    token = getpass.getpass(prompt='Please enter your Hugging Face token: ')
    login(token=token)
    print("Successfully logged in to Hugging Face!")

# Call the function to log in
login_to_huggingface()
```

- For detailed information on input files and parameters, see the [Input Documentation](input.md).
- To learn more about how to configure **scGenAI**, visit the [Configuration Guide](configuration.md).


_________


## Option 2: Running scGenAI from the Command Line

You can also run **scGenAI** directly from the command line by providing arguments for different modes (`Train`, `Predict`, `Finetune`). 

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


_________


##  Next Step: Output Files
Once the running is finished, the output model and files can be found in the path defined in the config file. Please read the doc of [Output Files](output.md) for details


