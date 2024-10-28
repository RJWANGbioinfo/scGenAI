import unittest
import os
import torch
from scgenai.config import Config, update_predconfig
from scgenai.finetune import FinetunePreprocessing, model_finetune_and_eval, postprocessing_summary
from scgenai.utils.load import *
from scgenai.utils.distributed import savelog

class TestFinetunePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests (called once for the entire test class)."""
        cls.config = update_predconfig("config_test_finetune.yaml")
        cls.config.savelog = "No"
        
        # Preprocess the finetuning data
        cls.adata_train, cls.adata_val, cls.train_loader, cls.val_loader, cls.custom_tokenizer, \
        cls.model_tokenizer, cls.le, cls.train_dataset, cls.model, cls.model_config, \
        cls.newlables, cls.newgenes = FinetunePreprocessing(cls.config)

    def test_finetune_preprocessing(self):
        """Test the preprocessing stage for fine-tuning."""
        self.assertIsNotNone(self.adata_train)
        self.assertIsNotNone(self.custom_tokenizer)
        self.assertEqual(len(self.adata_train.obs), len(self.train_loader.dataset))
    
    def test_finetune_model(self):
        """Test the actual fine-tuning step."""
        world_size = torch.cuda.device_count()
        rank = 0  # Simulate single GPU (rank 0) for the test

        # Run the fine-tuning on a single GPU for testing
        model_finetune_and_eval(
            rank, world_size, self.config, self.adata_train, self.adata_val,
            self.train_loader, self.train_dataset, self.val_loader, self.custom_tokenizer,
            self.le, self.model_tokenizer, self.model, self.model_config, self.newgenes, self.newlables
        )

        # Check if the model has been saved
        last_model_path = os.path.join(self.config.finetune_dir, "last_model", "pytorch_model.bin")
        self.assertTrue(os.path.exists(last_model_path), "Finetuned model not saved correctly.")

    def test_postprocessing(self):
        """Test the postprocessing after fine-tuning."""
        world_size = torch.cuda.device_count()

        # Run postprocessing for fine-tuning
        postprocessing_summary(world_size, self.config, self.adata_val)

        # Check if the combined results are saved correctly
        combined_results_file = os.path.join(self.config.finetune_dir, 'final_results.csv')
        self.assertTrue(os.path.exists(combined_results_file), "Postprocessed results not saved.")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after tests (called once after all tests)."""
        # Remove any generated test files or outputs
        if os.path.exists(cls.config.finetune_dir):
            shutil.rmtree(cls.config.finetune_dir)
        combined_results_file = os.path.join(cls.config.finetune_dir, 'final_results.csv')
        if os.path.exists(combined_results_file):
            os.remove(combined_results_file)


if __name__ == "__main__":
    unittest.main()
