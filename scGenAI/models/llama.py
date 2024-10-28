import torch
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

        
class CustomLlamaForSequenceClassification(nn.Module):
    def __init__(self, config, class_x=None, class_y=None):
        """
        Custom LLaMA model for sequence classification.

        Args:
            config (LlamaConfig): The configuration for the LLaMA model.
        """
        super(CustomLlamaForSequenceClassification, self).__init__()
        # Load LLaMA model for causal language modeling
        self.transformer = LlamaForCausalLM(config)
        # Classification layer
        if class_x is None:
            class_x = config.hidden_size

        if class_y is None:
            class_y = config.num_labels
        self.classifier = nn.Linear(class_x, class_y)
        self.config = config

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        """
        Forward pass for the custom LLaMA classification model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask for padding tokens.
            position_ids (torch.Tensor, optional): Position IDs for the input tokens.
            labels (torch.Tensor, optional): Ground truth labels for classification.

        Returns:
            Tuple: (loss, logits) if labels are provided, otherwise logits.
        """
        # Forward pass through the LLaMA transformer model
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        # Get hidden states of all tokens (first element of the output)
        hidden_states = transformer_outputs[0]
        
        # Dynamically adjust the classifier layer if the hidden size changes (for flexibility)
        if hidden_states.size(-1) != self.classifier.in_features:
            self.classifier = nn.Linear(hidden_states.size(-1), self.config.num_labels).to(hidden_states.device)

        # Use the hidden state of the last token for classification
        logits = self.classifier(hidden_states[:, -1, :])

        # If labels are provided, compute the loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # Return both loss and logits during training, or only logits during inference
        return (loss, logits) if loss is not None else logits

    def resize_token_embeddings(self, vocab_size):
        """
        Resize the token embeddings of the model.

        Args:
            vocab_size (int): New vocabulary size.
        """
        self.transformer.resize_token_embeddings(vocab_size)

class LlamaModelInitializer:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", cache_dir=None):
        """
        Initialize LlamaModelInitializer with tokenizer and configuration.

        Args:
            model_name (str): The pre-trained model name or path.
            cache_dir (str): Directory where the model is cached.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
    def get_tokenizer(self):
        """
        Load the LLaMA tokenizer.

        Returns:
            AutoTokenizer: The loaded tokenizer for the LLaMA model.
        """
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
    
    @classmethod
    def get_model_config(cls, hidden_size, intermediate_size, num_heads, num_layers, vocab_size, max_length):
        """
        Create the LLaMA model configuration.

        Args:
            hidden_size (int): The hidden size of the model.
            intermediate_size (int): The intermediate size of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers in the model.
            vocab_size (int): The size of the tokenizer's vocabulary.
            max_length (int): The maximum sequence length.

        Returns:
            LlamaConfig: The configuration object for the LLaMA model.
        """
        return LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
            max_position_embeddings=max_length
        )