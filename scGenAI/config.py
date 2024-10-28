import os
import yaml
import argparse
import shutil
class Config:
    def __init__(self, config_file=None, args=None, savesetting="Yes"):
        # Define base directory as the package root (two levels up from this file)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # General settings
        self.model_dir = os.path.join(base_dir, "examples", "model")  # Default to 'examples/model/' in the package
        self.finetune_dir = os.path.join(base_dir, "examples", "finetune")  # Default to 'examples/model/' in the package
        self.train_file = os.path.join(base_dir, "examples", "data", "NA.h5ad")  # Default training file
        self.val_file = os.path.join(base_dir, "examples", "data", "NA.h5ad")      # Default validation file
        self.train_ADTfile = os.path.join(base_dir, "examples", "data", "NA.h5ad")  # Default training file
        self.val_ADTfile = os.path.join(base_dir, "examples", "data", "NA.h5ad")      # Default validation file
        self.cytofile = os.path.join(base_dir, "examples", "data", "NA.txt")      # Default genomic file
        self.gmtfile = os.path.join(base_dir, "examples", "data", "NA.txt")      # Default genomic file
        self.glstfile = os.path.join(base_dir, "examples", "data", "NA.txt")      # Default genomic file
        self.predict_file = os.path.join(base_dir, "examples", "data", "NA.h5ad")  # Default predict file
        self.predict_ADTfile = os.path.join(base_dir, "examples", "data", "NA.h5ad")  # Default predict_ADT file
        self.log_dir = os.path.join(base_dir, "examples", "logs")      # Default logging directory
        self.output_dir = os.path.join(base_dir, "examples", "output")  # Default output directory
        self.outputfile = os.path.join(base_dir, "examples", "data", "NA.csv")  # Default predict file
        self.seed = 1314521
        self.cache_dir = os.path.join(base_dir, "cache")  # Default or specify it manually
        
        
        # Mode setting
        self.mode = "Train" # either 'Train', 'Predict', 'Finetune'
        
        
        # Preprocessing settings
        self.min_cells = 50
        self.target_feature = "celltype"
        self.multiomics = "No"
        self.keepIntermediateFiles = "No"
        self.savelog = "No" # Yes or No
        self.savesetting = savesetting
        
        # Model settings
        self.model_backbone_name = "llama"  # either 'gpt', 'bigbird', 'llama' or 'scGenT'
        self.model_backbone_size = "normal"  # either 'small', 'normal' or 'large'
        self.context_method = "random"  # either 'random', 'genomic', 'biofounction', 'genelist'
        
        # Hard-coded parameters for now
        self.num_bins = 10
        self.optimizer = "AdamW"
        self.weight_decay = 0.01
        self.depth = 2
        
        # Distributed training settings
        self.master_addr = "localhost"
        self.master_port = "12355"

        # Tokenizer settings
        self.max_length = 1024
        

        # Training settings
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.num_epochs = 30
        self.world_size = 1  # Default to single GPU or CPU

        # Auto-set evaluate_during_training if val_file is provided
        self.evaluate_during_training = bool(self.val_file)

        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)

        # Override with command-line arguments if provided
        if args:
            self.override_with_args(args)

        # Validate file paths
        self.check_files_exist()

    def load_config(self, config_file):
        """Load configuration from a YAML file."""
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Override defaults with config file values
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Auto-set any parameters based on updated values
        self.auto_set_model_parameters()
        
        if self.savesetting=="Yes" and self.mode == "Train":
            shutil.copy(config_file, os.path.join(self.model_dir, 'train_setting.yaml'))
        if self.savesetting=="Yes" and self.mode == "Finetune":
            shutil.copy(config_file, os.path.join(self.finetune_dir, 'train_setting.yaml'))
            
    def override_with_args(self, args):
        """Override configuration values with command-line arguments."""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def auto_set_model_parameters(self):
        """Automatically adjust parameters based on the config values."""
        # Set hidden sizes based on the model_backbone_size
        if self.model_backbone_size == "normal":
            self.hidden_size = 768
            self.num_layers = 12
            self.num_heads = 12
            if self.model_backbone_name in [ 'llama', 'bigbird']:
                self.intermediate_size=3072
            
        elif self.model_backbone_size == "large":
            self.hidden_size = 1024
            self.num_layers = 24
            self.num_heads = 16
            if self.model_backbone_name in [ 'llama', 'bigbird']:
                self.intermediate_size=4096
                
        elif self.model_backbone_size == "small":
            self.hidden_size = 256
            self.num_layers = 6
            self.num_heads = 4
            if self.model_backbone_name in [ 'llama', 'bigbird']:
                self.intermediate_size=1024
        self.stride = int(self.max_length / 2)
        # Automatically set evaluate_during_training if validation file is provided
        if not os.path.exists(self.val_file):
            self.evaluate_during_training = False
        else:
            self.evaluate_during_training = True
        self.learning_rate = float(self.learning_rate)
        self.depth = int(self.depth)
        if self.mode == "Predict":
            self.target_feature = "NOTROUBLESHOOT"
    def check_files_exist(self):
        """Check if all necessary files exist."""
        if self.mode == "Train":
            required_files = [
                ('train_file', self.train_file),
                ('model_dir', self.model_dir)
            ]
            for name, path in required_files:
                if path and not os.path.exists(path):
                    raise FileNotFoundError(f"{name} not found at path: {path}")
                    
            if self.multiomics == "Yes":
                if not os.path.exists(self.train_ADTfile):
                    raise FileNotFoundError(f"train_ADTfile not found at path: {self.train_ADTfile}")
            if self.evaluate_during_training:
                if not os.path.exists(self.val_file):
                    raise FileNotFoundError(f"val_file not found at path: {self.val_file}")
                    if self.multiomics == "Yes":
                        if not os.path.exists(self.val_ADTfile):
                            raise FileNotFoundError(f"train_ADTfile not found at path: {self.val_ADTfile}")
            if self.context_method=="genomic":
                if not os.path.exists(self.cytofile):
                    raise FileNotFoundError(f"cytofile not found at path: {self.cytofile}")
                elif self.savesetting=="Yes":
                    shutil.copy(self.cytofile, os.path.join(self.model_dir, 'genomic_context'))
            if self.context_method=="biofounction":
                if not os.path.exists(self.gmtfile):
                    raise FileNotFoundError(f"gmtfile not found at path: {self.gmtfile}")
                elif self.savesetting=="Yes":
                    shutil.copy(self.gmtfile, os.path.join(self.model_dir, 'biofounction_context'))
            if self.context_method=="genelist":
                if not os.path.exists(self.glstfile):
                    raise FileNotFoundError(f"glstfile not found at path: {self.glstfile}")
        elif self.mode == "Predict":
            if not os.path.exists(self.predict_file):
                raise FileNotFoundError(f"predict file not found at path: {self.predict_file}")
                if self.multiomics == "Yes":
                    if not os.path.exists(self.predict_ADTfile):
                        raise FileNotFoundError(f"predict_ADTfile not found at path: {self.predict_ADTfile}")
        elif self.mode == "Finetune":
            if not os.path.exists(self.finetune_dir):
                raise FileNotFoundError(f"finetune_dir not found at path: {self.finetune_dir}")
                
    def save_config(self, file_path):
        """Save the current configuration to a YAML file, excluding non-existent files or directories."""
        config_dict = {}
        
        # Iterate over the attributes and add to config_dict if the file or directory exists
        for key, value in self.__dict__.items():
            if isinstance(value, str) and os.path.exists(value):
                config_dict[key] = value
            elif not isinstance(value, str):
                if key!='evaluate_during_training':
                    config_dict[key] = value

        # Write the filtered configuration to a YAML file
        with open(file_path, "w") as file:
            yaml.dump(config_dict, file)

def update_predconfig(preconfigfile_path):
    config = Config(config_file=preconfigfile_path)
    trainset = Config(config_file=os.path.join(config.model_dir, 'train_setting.yaml'), savesetting="No")
    config.max_length = trainset.max_length
    config.stride = trainset.stride
    config.batch_size = trainset.batch_size
    config.min_cells = trainset.min_cells
    config.model_backbone_name = trainset.model_backbone_name
    config.model_backbone_size = trainset.model_backbone_size
    config.context_method = trainset.context_method
    config.num_bins = trainset.num_bins
    config.depth = trainset.depth
    config.multiomics = trainset.multiomics
    config.learning_rate = trainset.learning_rate
    config.intermediate_size = trainset.intermediate_size
    if config.context_method=="genomic":
        config.cytofile=os.path.join(config.model_dir, 'genomic_context')
    elif config.context_method=="biofounction":
        config.gmtfile=os.path.join(config.model_dir, 'biofounction_context')
    return config
        
# Argument parsing logic
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="scGenAI Configuration")
    
    # General settings
    parser.add_argument("--outputconfig", type=str, help="output YAML configuration file")
    parser.add_argument("--model_dir", type=str, help="Directory to save/load the model")
    parser.add_argument("--finetune_dir", type=str, help="Directory for fine-tuning models")
    parser.add_argument("--train_file", type=str, help="Path to the training data file")
    parser.add_argument("--val_file", type=str, help="Path to the validation data file")
    parser.add_argument("--train_ADTfile", type=str, help="Path to the multiomics training ADT file")
    parser.add_argument("--val_ADTfile", type=str, help="Path to the multiomics validation ADT file")
    parser.add_argument("--cytofile", type=str, help="Path to the cytoband data file")
    parser.add_argument("--gmtfile", type=str, help="Path to the GMT file for biofunction context")
    parser.add_argument("--glstfile", type=str, help="Path to the gene list file")
    parser.add_argument("--predict_file", type=str, help="Path to the prediction data file")
    parser.add_argument("--predict_ADTfile", type=str, help="Path to the multiomics prediction ADT file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--outputfile", type=str, help="Path to save prediction results")
    parser.add_argument("--seed", type=int, default=1314521, help="Random seed for reproducibility")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache models")
    
    # Mode setting
    parser.add_argument("--mode", type=str, choices=["Train", "Predict", "Finetune"], help="Operating mode")
    
    # Preprocessing settings
    parser.add_argument("--min_cells", type=int, help="Minimum number of cells for filtering")
    parser.add_argument("--target_feature", type=str, help="Target feature for training or prediction")
    parser.add_argument("--multiomics", type=str, choices=["Yes", "No"], help="Use multiomics data")
    parser.add_argument("--keepIntermediateFiles", type=str, choices=["Yes", "No"], help="Keep intermediate files")
    parser.add_argument("--savelog", type=str, choices=["Yes", "No"], help="Save log file into log dir")
    parser.add_argument("--evaluate_during_training", type=lambda x: (str(x).lower() == 'true'), help="Evaluate the model during training (True or False)")
    
    # Model settings
    parser.add_argument("--model_backbone_name", type=str, choices=["gpt", "bigbird", "llama", "scGenT"], help="Model backbone name")
    parser.add_argument("--model_backbone_size", type=str, choices=["small", "normal", "large"], help="Model backbone size")
    parser.add_argument("--context_method", type=str, choices=["random", "genomic", "biofounction", "genelist"], help="Context method")
    
    # Tokenizer settings
    parser.add_argument("--max_length", type=int, help="Maximum sequence length for the tokenizer")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    
    # Distributed training settings
    parser.add_argument("--world_size", type=int, help="Number of GPUs to use for distributed training")
    parser.add_argument("--master_addr", type=str, help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, help="Master port for distributed training")
    
    # Extra hard-coded parameters
    parser.add_argument("--num_bins", type=int, help="Number of bins for tokenization")
    parser.add_argument("--optimizer", type=str, choices=["AdamW"], help="Optimizer to use")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the optimizer")
    parser.add_argument("--depth", type=int, help="Depth for genomic context")

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    config = Config(args=args)  # Initialize configuration with command-line arguments

    if args.outputconfig:
        config.save_config(args.outputconfig)  # Save configuration to YAML file if specified

