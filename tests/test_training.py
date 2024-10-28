import unittest
import torch
import os
from scGenAI.training.train import run_training_from_config, DataPreprocessing, model_train_and_eval
from scGenAI.config import Config
from unittest.mock import patch, MagicMock

class TestTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a sample config file
        cls.config_file_path = 'test_config.yaml'
        with open(cls.config_file_path, 'w') as file:
            file.write("""
            model_backbone_name: 'llama'
            model_backbone_size: 'small'
            max_length: 1024
            batch_size: 2
            num_epochs: 1
            train_file: '../examples/data/sample_train.h5ad'
            val_file: '../examples/data/sample_val.h5ad'
            target_feature: 'celltype'
            model_dir: '../examples/output/test'
            """)

    @classmethod
    def tearDownClass(cls):
        # Remove config file after tests
        if os.path.exists(cls.config_file_path):
            os.remove(cls.config_file_path)
    
    @patch('scGenAI.training.train.DataPreprocessing')
    def test_data_preprocessing(self, mock_preprocessing):
        config = Config(config_file=self.config_file_path)
        mock_preprocessor_instance = MagicMock()
        mock_preprocessing.return_value = mock_preprocessor_instance
        
        # Simulate return values for preprocessing
        mock_preprocessor_instance.preprocess_data.return_value = (None, None, [], [])
        adata_train, adata_val, train_loader, val_loader, custom_tokenizer, model_tokenizer, le, train_dataset = DataPreprocessing(config)
        
        # Check if preprocess_data was called
        mock_preprocessor_instance.preprocess_data.assert_called_once()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(custom_tokenizer)
    
    @patch('scGenAI.training.train.model_train_and_eval')
    @patch('torch.multiprocessing.spawn')
    def test_training_invocation(self, mock_multiprocessing, mock_train_eval):
        config = Config(config_file=self.config_file_path)

        # Mock torch multiprocessing and train/eval
        mock_multiprocessing.return_value = None
        mock_train_eval.return_value = None

        run_training_from_config(self.config_file_path)
        
        # Ensure model_train_and_eval gets called
        mock_multiprocessing.assert_called_once()
        mock_train_eval.assert_not_called()  # Because multiprocessing.spawn is mocked

    def test_model_initialization(self):
        config = Config(config_file=self.config_file_path)
        model_backbone_name = 'llama'
        
        # Ensure the right model initializer is used
        if model_backbone_name == 'llama':
            from scGenAI.models.llama import LlamaModelInitializer
            model_initializer = LlamaModelInitializer(cache_dir=config.cache_dir)
            model_config = model_initializer.get_model_config(256, 4, 6, 10, 1024, 30000)
            self.assertIsNotNone(model_config)

    def test_train_and_evaluate(self):
        config = Config(config_file=self.config_file_path)
        world_size = torch.cuda.device_count()

        # Ensure single GPU or CPU is called if no multiprocessing
        with patch('scGenAI.training.train.model_train_and_eval') as mock_train_eval:
            mock_train_eval.return_value = None
            model_train_and_eval(0, world_size, config, None, None, None, None, None, None, None)
            mock_train_eval.assert_called_once()

if __name__ == '__main__':
    unittest.main()
