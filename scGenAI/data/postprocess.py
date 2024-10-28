import os
import pandas as pd
import torch
from loguru import logger
import shutil
import matplotlib.pyplot as plt

class Postprocessor:
    def __init__(self, model_dir, target_name, world_size, outputfile=None, adata_pre=None, keepIntermediateFiles="No"):
        """
        Initialize the Postprocessor class.

        Args:
            model_dir (str): Path to the directory where model outputs are stored.
            target_name (str): Name of the target feature in the validation dataset (e.g., 'celltype').
            world_size (int): Number of distributed processes (GPUs) used for training.
        """
        self.model_dir = model_dir
        self.target_name = target_name
        self.world_size = world_size
        self.outputfile = outputfile
        self.adata_pre = adata_pre
        self.keepIntermediateFiles = keepIntermediateFiles
        
    def combine_epoch_results(self):
        """
        Combine epoch_losses.txt, epoch_accuracies.txt, and epoch_summary.csv into one CSV file, 
        then remove these files.
        """
        # File paths
        losses_file = os.path.join(self.model_dir, 'epoch_losses.txt')
        accuracies_file = os.path.join(self.model_dir, 'epoch_accuracies.txt')
        summary_file = os.path.join(self.model_dir, 'epoch_summary.csv')

        # Read the loss and accuracy files as lists
        with open(losses_file, 'r') as f:
            losses = [line.strip() for line in f.readlines()[1:]]  # Skip the header
        with open(accuracies_file, 'r') as f:
            accuracies = [line.strip() for line in f.readlines()[1:]]  # Skip the header

        # Read the summary CSV
        summary_df = pd.read_csv(summary_file)

        # Ensure that the lengths of all data match
        if len(losses) != len(accuracies) or len(losses) != len(summary_df):
            raise ValueError("Mismatch in number of epochs between losses, accuracies, and summary data.")

        # Add losses and accuracies to the summary DataFrame
        summary_df['Train Loss'] = losses
        summary_df['Train Accuracy'] = accuracies

        # Save the combined DataFrame
        combined_file = os.path.join(self.model_dir, 'combined_epoch_results.csv')
        summary_df.to_csv(combined_file, index=False)

        # Remove the old files
        os.remove(losses_file)
        os.remove(accuracies_file)
        os.remove(summary_file)

        logger.info(f"Combined results saved to {combined_file}.")

    def combine_trainonly_results(self):
        """
        Combine epoch_losses.txt, epoch_accuracies.txt, 
        then remove these files.
        """
        # File paths
        losses_file = os.path.join(self.model_dir, 'epoch_losses.txt')
        accuracies_file = os.path.join(self.model_dir, 'epoch_accuracies.txt')

        # Read the loss and accuracy files as lists
        with open(losses_file, 'r') as f:
            losses = [line.strip() for line in f.readlines()[1:]]  # Skip the header
        with open(accuracies_file, 'r') as f:
            accuracies = [line.strip() for line in f.readlines()[1:]]  # Skip the header
        if len(losses) != len(accuracies):
            raise ValueError("Mismatch in number of epochs between losses and accuracies")
        summary_df = pd.DataFrame({'Train Loss': losses, 'Train Accuracy': accuracies})
        summary_df['Epoch'] = summary_df.index.copy() + 1

        # Save the combined DataFrame
        combined_file = os.path.join(self.model_dir, 'combined_epoch_results.csv')
        summary_df.to_csv(combined_file, index=False)

        # Remove the old files
        os.remove(losses_file)
        os.remove(accuracies_file)
        logger.info(f"Combined results saved to {combined_file} and intermediate files removed.")

    def combine_prediction_results(self):
        final_results_df = torch.load(os.path.join(self.model_dir, f"results_df_rank_0.pt"))
        os.remove(os.path.join(self.model_dir, f"results_df_rank_0.pt"))
        for i in range(1, self.world_size):
            temp_results_df = torch.load(os.path.join(self.model_dir, f"results_df_rank_{i}.pt"))
            final_results_df = pd.concat([final_results_df, temp_results_df], ignore_index=True)
            os.remove(os.path.join(self.model_dir, f"results_df_rank_{i}.pt"))

        final_predictions_df = final_results_df.loc[final_results_df.groupby('cell_barcode')['prediction_score'].idxmax()]
        final_predictions_df.rename(columns={'prediction': 'PredictedFeature'}, inplace=True)
        merged_df = pd.merge(self.adata_pre.obs.reset_index(), final_predictions_df, left_on='index', right_on='cell_barcode', how='outer')
        merged_df.to_csv(self.outputfile, index=False)
        logger.info("Completed post-processing for prediction")
    
    def move_intermediate_files(self):
        """
        Move all 'merged_final_predictions*' and 'results_df_epoch*' files into a subfolder called 'IntermediateFiles'.
        """
        intermediate_folder = os.path.join(self.model_dir, 'IntermediateFiles')

        # save gene files for multiomics
        RNAfile = os.path.join(self.model_dir, 'trainedRNA_genes.npy')
        ADTfile = os.path.join(self.model_dir, 'gene_list_to_emphasize.npy')
        cytofile = os.path.join(self.model_dir, 'genomic_context')
        gmtfile = os.path.join(self.model_dir, 'biofounction_context')
        ymlfile = os.path.join(self.model_dir, 'train_setting.yaml')
        for file in [RNAfile, ADTfile, cytofile, gmtfile, ymlfile]:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(self.model_dir, 'best_model/'))
                shutil.copy(file, os.path.join(self.model_dir, 'last_model/'))
                os.remove(file)
        if self.keepIntermediateFiles!="No":
            # Create IntermediateFiles folder if it doesn't exist
            os.makedirs(intermediate_folder, exist_ok=True)

                
        # List and move all relevant files
        for file_name in os.listdir(self.model_dir):
            if file_name.startswith('merged_final_predictions') or file_name.startswith('results_df_epoch') or file_name.startswith('context_matrix_'):
                if self.keepIntermediateFiles!="No":
                    old_path = os.path.join(self.model_dir, file_name)
                    new_path = os.path.join(intermediate_folder, file_name)
                    shutil.move(old_path, new_path)
                else:
                    os.remove(os.path.join(self.model_dir, file_name))

    def merge_predictions_across_gpus(self, epoch):
        """
        Merge predictions from multiple GPUs (distributed processes) into a single DataFrame for a given epoch.

        Args:
            epoch (int): The epoch number for which predictions need to be merged.

        Returns:
            final_results_df (pd.DataFrame): Merged DataFrame containing predictions from all GPUs.
        """
        logger.info(f"Merging predictions for epoch {epoch+1}")
        final_results_df = torch.load(os.path.join(self.model_dir, f"results_df_epoch_{epoch}_rank_0.pt"))

        # Load and concatenate results from other processes
        for rank in range(1, self.world_size):
            temp_results_df = torch.load(os.path.join(self.model_dir, f"results_df_epoch_{epoch}_rank_{rank}.pt"))
            final_results_df = pd.concat([final_results_df, temp_results_df], ignore_index=True)
        
        return final_results_df

    def select_best_predictions(self, final_results_df):
        """
        Select the best predictions for each cell based on the highest prediction score.

        Args:
            final_results_df (pd.DataFrame): DataFrame containing prediction results for each window.

        Returns:
            final_predictions_df (pd.DataFrame): DataFrame with the best prediction per cell.
        """
        # Get the prediction with the highest score for each cell
        final_predictions_df = final_results_df.loc[final_results_df.groupby('cell_barcode')['prediction_score'].idxmax()]
        final_predictions_df.rename(columns={'prediction': 'PredictedFeature'}, inplace=True)
        return final_predictions_df

    def calculate_val_accuracy(self, adata_val, final_predictions_df):
        """
        Calculate the validation accuracy by comparing true labels with predicted labels.

        Args:
            adata_val (AnnData): AnnData object containing the validation data.
            final_predictions_df (pd.DataFrame): DataFrame containing the best predictions per cell.

        Returns:
            val_accuracy (float): Validation accuracy for the current epoch.
            merged_df (pd.DataFrame): Merged DataFrame containing true labels and predictions.
        """
        # Merge true labels from validation data with predictions
        merged_df = pd.merge(
            adata_val.obs.reset_index(), final_predictions_df,
            left_on='index', right_on='cell_barcode', how='outer'
        )

        # Compute validation accuracy by comparing true and predicted labels
        val_accuracy = (merged_df[self.target_name] == merged_df['PredictedFeature']).mean()

        return val_accuracy, merged_df

    def postprocess_epoch(self, epoch, adata_val):
        """
        Perform postprocessing for a single epoch: merge predictions, select best predictions, and calculate accuracy.

        Args:
            epoch (int): The epoch number to postprocess.
            adata_val (AnnData): AnnData object containing validation data.

        Returns:
            val_accuracy (float): Validation accuracy for the current epoch.
            merged_df (pd.DataFrame): DataFrame with true labels and predicted labels.
        """
        final_results_df = self.merge_predictions_across_gpus(epoch)
        final_predictions_df = self.select_best_predictions(final_results_df)
        val_accuracy, merged_df = self.calculate_val_accuracy(adata_val, final_predictions_df)
        merged_df.to_csv(os.path.join(self.model_dir, f'merged_final_predictions_epoch_{epoch + 1}.csv'), index=False)
        logger.info(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.4f}")
        return val_accuracy

    def summarize_epochs(self, adata_val, num_epochs):
        """
        Summarize the results across all epochs by calculating validation accuracy for each epoch.

        Args:
            adata_val (AnnData): AnnData object containing validation data.
            num_epochs (int): Total number of epochs.

        Returns:
            summary_df (pd.DataFrame): DataFrame summarizing the validation accuracy for each epoch.
        """
        epoch_summary = []

        for epoch in range(num_epochs):
            val_accuracy = self.postprocess_epoch(epoch, adata_val)
            epoch_summary.append((epoch + 1, val_accuracy))

        # Save the summary as a CSV
        summary_df = pd.DataFrame(epoch_summary, columns=["Epoch", "Validation Accuracy"])
        summary_df.to_csv(os.path.join(self.model_dir, 'epoch_summary.csv'), index=False)
        logger.info("Completed post-processing and summary of all epochs.")

        return summary_df

class TrainSummary:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.results_file = os.path.join(self.model_dir, 'combined_epoch_results.csv')
    
    def table(self):
        """
        Reads the 'combined_epoch_results.csv' file from the model directory 
        and returns it as a pandas DataFrame.
        """
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"{self.results_file} does not exist.")
        df = pd.read_csv(self.results_file)
        return df

    def plot(self, saveplot=False):
        """
        Reads the 'combined_epoch_results.csv' file and plots the Train Accuracy, Train Loss,
        and optionally Validation Accuracy if it exists in the file.
        """
        df = self.table()  # Reuse the table function to get the dataframe
        
        # Check if necessary columns exist
        if 'Epoch' not in df.columns or 'Train Accuracy' not in df.columns or 'Train Loss' not in df.columns:
            raise ValueError("The file does not contain necessary columns for plotting.")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train Accuracy'], 'o-', label='Train Accuracy', color='blue')
        plt.plot(df['Epoch'], df['Train Loss'], 'o-', label='Train Loss', color='red')

        if 'Validation Accuracy' in df.columns:
            plt.plot(df['Epoch'], df['Validation Accuracy'], 'o-', label='Validation Accuracy', color='green')

        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Training and Validation Metrics Over Epochs')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PDF if saveplot=True
        if saveplot:
            save_path = os.path.join(self.model_dir, 'train_summary.pdf')
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            