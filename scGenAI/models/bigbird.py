import torch
from torch import nn
from transformers import BigBirdTokenizer, BigBirdConfig, BigBirdModel


class CustomBigBirdForSequenceClassification(nn.Module):
    def __init__(self, config, class_x=None, class_y=None):
        """
        Custom model for sequence classification using BigBird.
        
        Args:
            config (BigBirdConfig): Configuration for the BigBird model.
        """
        super(CustomBigBirdForSequenceClassification, self).__init__()
        self.transformer = BigBirdModel(config)
        if class_x is not None and class_y is not None:
            self.classifier = nn.Linear(class_x, class_y)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        """
        Forward pass for BigBird classification.
        
        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Masking for attention (optional).
            position_ids (Tensor): Position IDs (optional).
            labels (Tensor): Target labels for classification (optional).
            
        Returns:
            Tuple of (loss, logits) if labels are provided; otherwise, logits.
        """
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.classifier(hidden_states[:, -1, :])  # Use the last token's hidden state
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

    def resize_token_embeddings(self, vocab_size):
        """
        Resize the token embeddings of the model.

        Args:
            vocab_size (int): New vocabulary size.
        """
        self.transformer.resize_token_embeddings(vocab_size)


class BigBirdModelInitializer:
    def __init__(self, model_name="google/bigbird-roberta-base", cache_dir=None):
        """
        Initialize BigBird model and tokenizer.
        
        Args:
            model_name (str): Name of the pretrained BigBird model.
            cache_dir (str): Directory to cache the model.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
    def get_tokenizer(self):
        """
        Load the BigBird tokenizer.
        
        Returns:
            BigBirdTokenizer: Tokenizer for the BigBird model.
        """
        return BigBirdTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    @classmethod
    def get_model_config(cls, hidden_size, num_attention_heads, num_hidden_layers, vocab_size, num_labels, max_position_embeddings, intermediate_size):
        """
        Create a custom BigBird model configuration.
        
        Args:
            hidden_size (int): The hidden size of the model.
            intermediate_size (int): The intermediate size of the model.
            num_attention_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of layers in the model.
            vocab_size (int): The size of the tokenizer's vocabulary.
            max_position_embeddings (int): The maximum sequence length.
        
        Returns:
            BigBirdConfig: Configuration object for BigBird.
        """
        return BigBirdConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_labels=num_labels,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,  # Adjust according to your needs
            attention_probs_dropout_prob=0.1,  # Standard dropout
            hidden_dropout_prob=0.1,
            use_cache=False  # Disable caching to save memory
        )
