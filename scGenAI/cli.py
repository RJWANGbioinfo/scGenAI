import argparse
from .training.train import run_training_from_config
from .prediction.predict import run_prediction_from_config
from .finetuning.finetune import run_finetune_from_config

def main():
    parser = argparse.ArgumentParser(description="scGenAI Command Line Interface")
    
    # Subparsers for the different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands: train, predict, finetune")
    
    # Train command
    parser_train = subparsers.add_parser("train", help="Run training")
    parser_train.add_argument("--config_file", type=str, required=True, help="Path to the training configuration file")
    
    # Predict command
    parser_predict = subparsers.add_parser("predict", help="Run prediction")
    parser_predict.add_argument("--config_file", type=str, required=True, help="Path to the prediction configuration file")
    
    # Finetune command
    parser_finetune = subparsers.add_parser("finetune", help="Run finetuning")
    parser_finetune.add_argument("--config_file", type=str, required=True, help="Path to the finetune configuration file")
    
    args = parser.parse_args()

    # Dispatch to the correct function based on the command
    if args.command == "train":
        run_training_from_config(args.config_file)
    elif args.command == "predict":
        run_prediction_from_config(args.config_file)
    elif args.command == "finetune":
        run_finetune_from_config(args.config_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
