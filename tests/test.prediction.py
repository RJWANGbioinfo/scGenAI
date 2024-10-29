import unittest
import os
import torch
from scgenai.config import Config
from scgenai.utils.load import *
from scgenai.predict import PredictionPreprocessing, Prediction, PostprocessingPrediction

class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Set up shared resources for all tests (called once for the entire test class). """
        cls.config = Config(config_file="config_test_pred.yaml")
        cls.config.savelog = "No"

        # Preprocess the prediction data
        cls.adata_pre, cls.pre_sequences, cls.tokenized_pre_sequences, cls.barcodes_pre, cls.custom_tokenizer, cls.y_encoded_pre, cls.class_x, cls.class_y, cls.le = PredictionPreprocessing(cls.config)

    def test_prediction_preprocessing(self):
        """ Test the preprocessing stage. """
        self.assertIsNotNone(self.adata_pre)
        self.assertIsNotNone(self.tokenized_pre_sequences)
        self.assertEqual(len(self.barcodes_pre), len(self.adata_pre.obs.index))

    def test_prediction_run(self):
        """ Test the actual prediction step. """
        world_size = torch.cuda.device_count()
        rank = 0  # We simulate single GPU (rank 0) for this test

        # Call prediction on rank 0 for testing (you may want to use smaller data)
        Prediction(rank, self.adata_pre, self.pre_sequences, self.tokenized_pre_sequences, self.barcodes_pre, 
                   self.custom_tokenizer, world_size, self.y_encoded_pre, self.le, self.class_x, self.class_y, self.config)

        # Check if results were saved
        results_file = os.path.join(self.config.model_dir, f"results_df_rank_{rank}.pt")
        self.assertTrue(os.path.exists(results_file), "Results file was not saved.")

    def test_postprocessing(self):
        """ Test the postprocessing of prediction results. """
        world_size = torch.cuda.device_count()

        # Perform postprocessing
        PostprocessingPrediction(world_size, self.config, self.adata_pre)

        # Check the final output file
        output_file = self.config.outputfile
        self.assertTrue(os.path.exists(output_file), "Final output file was not created after postprocessing.")

    @classmethod
    def tearDownClass(cls):
        """ Clean up resources after tests (called once after all tests). """
        # Remove any test files or outputs generated
        if os.path.exists(cls.config.model_dir):
            shutil.rmtree(cls.config.model_dir)
        if os.path.exists(cls.config.outputfile):
            os.remove(cls.config.outputfile)

if __name__ == "__main__":
    unittest.main()
