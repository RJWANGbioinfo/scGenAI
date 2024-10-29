import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model

class CustomGPT2ForSequenceClassification(nn.Module):
    def __init__(self, config, class_x=None, class_y=None):
        super(CustomGPT2ForSequenceClassification, self).__init__()
        self.transformer = GPT2Model(config)
        if class_x is not None and class_y is not None:
            self.classifier = nn.Linear(class_x, class_y)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.classifier(hidden_states[:, -1, :])
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

class GPTModelInitializer:
    def __init__(self, model_name="gpt2", cache_dir=None):
        """
        Initialize GPTModelInitializer with tokenizer and configuration.

        Args:
            model_name (str): The pre-trained model name or path.
            cache_dir (str): Directory where the model is cached.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
    def get_tokenizer(self):
        """
        Load the GPT tokenizer.

        Returns:
            GPT2Tokenizer: The loaded tokenizer for the GPT model.
        """
        return GPT2Tokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    @classmethod
    def get_model_config(cls, n_embd, num_heads, num_layers, vocab_size, num_labels):
        """
        Load the GPT model configuration.

        Args:
            n_embd (int): The hidden size (embedding dimension) of the model.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of hidden layers.
            vocab_size (int): The size of the vocabulary.
            num_labels (int): The number of output labels for classification.

        Returns:
            GPT2Config: The GPT2 model configuration.
        """
        return GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,  # Hidden size (embedding dimension)
            n_head=num_heads,
            n_layer=num_layers,
            num_labels=num_labels,
            initializer_range=0.02,  # Weight initialization range
            use_cache=False  # Disable cache to save memory
        )
