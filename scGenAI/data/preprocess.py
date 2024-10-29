import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import anndata as ad
import warnings, os
from ..utils.load import loadRNAADTgenes

warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")

        

class GeneExpressionTokenizer:
    def __init__(self, base_tokenizer, gene_names, bins=50):
        bins = bins+1
        self.base_tokenizer = base_tokenizer
        self.gene_vocab = {gene: i + self.base_tokenizer.vocab_size for i, gene in enumerate(gene_names)}
        self.expression_vocab = {str(i): i + self.base_tokenizer.vocab_size + len(gene_names) for i in range(1, bins + 1)}
        self.vocab_size = self.base_tokenizer.vocab_size + len(self.gene_vocab) + len(self.expression_vocab) + 1
        self.pad_token_id = self.vocab_size - 1

    def encode(self, genes, expressions):
        tokens = []
        for gene, expr in zip(genes, expressions):
            gene_id = self.gene_vocab.get(gene, self.pad_token_id)
            expr_id = self.expression_vocab.get(expr, self.pad_token_id)
            tokens.append(gene_id)
            tokens.append(expr_id)
        return tokens

    def __call__(self, sequences):
        encoded_sequences = []
        for genes, expressions in sequences:
            encoded_sequences.append(self.encode(genes, expressions))
        return encoded_sequences

class Preprocessor:
    def __init__(self, train_file="NA", val_file="NA", train_ADTfile="NA", val_ADTfile="NA", predict_file="NA", predict_ADTfile="NA", model_dir="NA", trained_genes=None, min_cells=50, seed=42, max_length=1024, bins=50, subset="No", glstfile="NA"):
        """
        Initialize the Preprocessor class with necessary parameters.

        Args:
            train_file (str): Path to the training data (.h5ad).
            val_file (str): Path to the validation data (.h5ad).
            min_cells (int): Minimum number of cells to filter genes.
            seed (int): Random seed for shuffling genes.
            max_length (int): Maximum sequence length for the model.
        """
        self.train_file = train_file
        self.val_file = val_file
        self.train_ADTfile = train_ADTfile
        self.val_ADTfile = val_ADTfile 
        self.predict_file = predict_file
        self.predict_ADTfile = predict_ADTfile
        self.val_ADTfile = val_ADTfile 
        self.min_cells = min_cells
        self.seed = seed
        self.max_length = max_length
        self.trained_genes = trained_genes
        self.subset = subset
        self.glstfile = glstfile
        self.model_dir = model_dir
        self.bins = int(bins)
        self.adata_train = None
        self.adata_val = None
        self.adata_ADTtrain = None
        self.adata_ADTval = None
        self.adata_pre = None
        self.adata_ADTpre = None
        

    def load_data(self):
        """Load the training and validation data from .h5ad files."""
        if os.path.exists(self.train_file):
            self.adata_train = sc.read_h5ad(self.train_file)
        if os.path.exists(self.val_file):
            self.adata_val = sc.read_h5ad(self.val_file)
        if os.path.exists(self.train_ADTfile):
            self.adata_ADTtrain = sc.read_h5ad(self.train_ADTfile)
        if os.path.exists(self.val_ADTfile):
            self.adata_ADTval = sc.read_h5ad(self.val_ADTfile)
        if os.path.exists(self.predict_file):
            self.adata_pre = sc.read_h5ad(self.predict_file)
        if os.path.exists(self.predict_ADTfile):
            self.adata_ADTpre = sc.read_h5ad(self.predict_ADTfile)

    def filter_genes(self, adata):
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        return adata
        
    def shuffle_genes(self, adata):
        np.random.seed(self.seed)
        shuffled_gene_indices = np.random.permutation(adata.var_names)
        adata = adata[:, shuffled_gene_indices]
        return adata
        
    def filter_and_shuffle_genes(self):
        """Filter genes based on cell counts and shuffle gene order."""
        self.adata_train = self.filter_genes(self.adata_train)
        self.adata_train = self.shuffle_genes(self.adata_train)

    def normalize_and_log_transform(self, adata):
        """Normalize total counts per cell and apply log transformation."""
        if adata.X.max() > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        return adata
        
    def bin_expression_values(self, adata, num_bins=10):
        """
        Bin the expression values into discrete categories for tokenization.

        Args:
            adata: AnnData object containing gene expression data.
            num_bins: Number of bins to discretize expression values.

        Returns:
            binned_data: Numpy array of binned expression values.
        """
        
        num_bins = num_bins+1
        data_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        binned_data = np.zeros_like(data_dense)

        for i, cell in enumerate(data_dense):
            non_zero_indices = np.nonzero(cell)[0]
            if len(non_zero_indices) > 0:
                non_zero_values = cell[non_zero_indices]
                binned_values = np.digitize(non_zero_values, bins=np.linspace(1, non_zero_values.max(), num_bins))
                binned_values = np.clip(binned_values, 1, num_bins)
                binned_data[i, non_zero_indices] = binned_values
        return binned_data
        
    def align_genes(self, adata2, trained_genes):
        genes_adata = pd.Index(trained_genes)
        genes_adata2 = adata2.var_names
        common_genes = genes_adata.intersection(genes_adata2)
        adata2_aligned = adata2[:, common_genes]
        missing_genes = genes_adata.difference(genes_adata2)
        if len(missing_genes) > 0:
            zero_data = np.zeros((adata2_aligned.shape[0], len(missing_genes)))
            missing_df = pd.DataFrame(zero_data, columns=missing_genes, index=adata2_aligned.obs_names)
            missing_adata = sc.AnnData(X=missing_df, var=pd.DataFrame(index=missing_genes), obs=adata2_aligned.obs)
            adata2_aligned = sc.concat([adata2_aligned, missing_adata], axis=1)
            adata2_aligned = adata2_aligned[:, genes_adata]
            adata2_aligned.obs=adata2.obs.copy()
        return adata2_aligned
    
    def align_genes_val(self, trained_genes):
        self.adata_val = self.align_genes(self.adata_val, trained_genes)

    def convert_to_sequences(self, adata):
        """
        Convert the gene expression data to sequences of genes and expression levels.

        Args:
            adata: AnnData object containing gene expression data.

        Returns:
            sequences: List of tuples (genes, expression_levels) for each cell.
        """
        sequences = []
        data_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        for cell in data_dense:
            non_zero_indices = np.nonzero(cell)[0]
            genes = adata.var_names[non_zero_indices]
            expressions = [str(int(value)) for value in cell[non_zero_indices]]
            sequences.append((genes, expressions))

        return sequences
    
    def subsetbygenelist(self, adata):
        with open(self.glstfile, 'r') as f:
            gene_list = [line.strip() for line in f]
        genes_in_adata = [gene for gene in gene_list if gene in adata.var_names]
        adata = adata[:, genes_in_adata]
        return adata
        
    def extractgene_lable(self, targetname):
        self.load_data()
        self.filter_and_shuffle_genes()
        return self.adata_train.var_names.unique().tolist(), self.adata_train.obs[targetname].unique().tolist()

    def extractmultiomicsgene_lable(self, targetname):
        self.load_data()
        self.filter_and_shuffle_genes()
        self.adata_ADTtrain = self.filter_genes(self.adata_ADTtrain)
        self.adata_ADTtrain = self.shuffle_genes(self.adata_ADTtrain)
        return self.adata_train.var_names.unique().tolist(), self.adata_ADTtrain.var_names.unique().tolist(), self.adata_train.obs[targetname].unique().tolist()

        
    def preprocess_data(self, finetune_genes=None):
        """
        Perform the complete preprocessing pipeline: loading, filtering, shuffling, binning,
        and aligning genes for both training and validation datasets.

        Returns:
            adata_train: Preprocessed AnnData object for training data.
            adata_val: Preprocessed AnnData object for validation data.
            train_sequences: List of tokenized training sequences.
            val_sequences: List of tokenized validation sequences.
        """
        # Load the data
        self.load_data()
        

            
        # Filter and shuffle genes in training data
        self.filter_and_shuffle_genes()

        if finetune_genes is not None:
            self.adata_train = self.align_genes(self.adata_train, finetune_genes)
            
        if self.subset=="genelist":
            self.adata_train = self.subsetbygenelist(self.adata_train)
        # Normalize and log-transform the training data
        self.adata_train = self.normalize_and_log_transform(self.adata_train)

        # Bin expression values for training data
        binned_data_train = self.bin_expression_values(self.adata_train, num_bins=self.bins)
        self.adata_train.X = binned_data_train

        # Convert data to sequences
        train_sequences = self.convert_to_sequences(self.adata_train)
        
        if self.adata_val is not None:
            # Align genes in validation data
            self.align_genes_val(self.adata_train.var_names)

            # Normalize and log-transform validation data
            self.adata_val = self.normalize_and_log_transform(self.adata_val)

            # Bin expression values for validation data
            binned_data_val = self.bin_expression_values(self.adata_val, num_bins=self.bins)
            self.adata_val.X = binned_data_val
            val_sequences = self.convert_to_sequences(self.adata_val)
        else:
            val_sequences = None
        return self.adata_train, self.adata_val, train_sequences, val_sequences
    
    def dataalignfile(self, adata, seed=1314521, trained_genes=None, num_bins=11, finetune_genes=None):
        if trained_genes is not None:
            adata = self.align_genes(adata, trained_genes)
        else:
            adata = self.shuffle_genes(adata)
        if trained_genes is None:
            adata = self.filter_genes(adata)
        if finetune_genes is not None:
            adata = self.align_genes(adata, finetune_genes)
        adata = self.normalize_and_log_transform(adata)
        binned_adata = self.bin_expression_values(adata, num_bins=num_bins)
        adata.X = binned_adata
        return adata
        
    def preprocess_multiomicsdata(self, finetune_genes=None, finetune_adts=None):
        # Load the data
        self.load_data()
        # process train data
        adata_trainRNA = self.dataalignfile(self.adata_train, seed=self.seed, trained_genes=None, num_bins=self.bins, finetune_genes=finetune_genes)
        adata_trainADT = self.dataalignfile(self.adata_ADTtrain, seed=self.seed, trained_genes=None, num_bins=6, finetune_genes=finetune_adts)
        adata_trainADT = adata_trainADT[adata_trainRNA.obs_names, :]
        self.adata_train = ad.concat([adata_trainRNA, adata_trainADT], axis=1)
        self.adata_train.obs = adata_trainRNA.obs.copy()
        gene_list_to_emphasize = adata_trainADT.var_names.tolist()
        gene_in_trainRNA = adata_trainRNA.var_names.tolist()
        np.save(os.path.join(self.model_dir, 'trainedRNA_genes.npy'), gene_in_trainRNA)
        np.save(os.path.join(self.model_dir, 'gene_list_to_emphasize.npy'), gene_list_to_emphasize)
        
        if self.subset=="genelist":
            self.adata_train = self.subsetbygenelist(self.adata_train)
        # Convert data to sequences
        train_sequences = self.convert_to_sequences(self.adata_train)
        
        if self.adata_val is not None:
            adata_valRNA = self.dataalignfile(self.adata_val, seed=self.seed, trained_genes=gene_in_trainRNA, num_bins=self.bins)
            adata_valADT = self.dataalignfile(self.adata_ADTval, seed=self.seed, trained_genes=gene_list_to_emphasize, num_bins=6)
            adata_valADT = adata_valADT[adata_valRNA.obs_names, :]
            self.adata_val = ad.concat([adata_valRNA, adata_valADT], axis=1)
            self.adata_val.obs = adata_valRNA.obs.copy()
            val_sequences = self.convert_to_sequences(self.adata_val)
        else:
            val_sequences = None
        return self.adata_train, self.adata_val, train_sequences, val_sequences, gene_list_to_emphasize

    def preprocess_prediction(self):
        self.load_data()
        self.adata_pre = self.align_genes(self.adata_pre, self.trained_genes)
        self.adata_pre = self.normalize_and_log_transform(self.adata_pre)
        binned_adata = self.bin_expression_values(self.adata_pre, num_bins=self.bins)
        self.adata_pre.X = binned_adata
        
        pre_sequences = self.convert_to_sequences(self.adata_pre)
        
        return self.adata_pre, pre_sequences
        
    def preprocess_multiomicsprediction(self):
        self.load_data()
        gene_in_trainRNA, gene_list_to_emphasize = loadRNAADTgenes(self.model_dir)
        adata_preRNA = self.dataalignfile(self.adata_pre, seed=self.seed, trained_genes=gene_in_trainRNA.tolist(), num_bins=self.bins)
        adata_preADT = self.dataalignfile(self.adata_ADTpre, seed=self.seed, trained_genes=gene_list_to_emphasize.tolist(), num_bins=5)
        adata_preADT = adata_preADT[adata_preRNA.obs_names, :]
        self.adata_pre = ad.concat([adata_preRNA, adata_preADT], axis=1)
        self.adata_pre.obs = adata_preRNA.obs.copy()
        pre_sequences = self.convert_to_sequences(self.adata_pre)
        
        return self.adata_pre, pre_sequences, gene_list_to_emphasize