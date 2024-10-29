import os
import torch
import json
from torch import nn
from collections import defaultdict, Counter
import math

# Define the maximum sequence length as a parameter
MAX_SEQ_LENGTH = 2048

class BPETokenizer:
    def __init__(self, vocab_size, eos_token="</s>"):
        self.vocab_size = vocab_size
        self.bpe_ranks = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        self.eos_token = eos_token
        self.eos_token_id = self.vocab_size
        
    def train(self, corpus):
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]
        vocab = Counter(tokens)

        bpe_merges = []
        while len(vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            bpe_merges.append(best)
            vocab = self.merge_vocab(best, vocab)

        self.bpe_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        self.vocab = vocab
        self.token_to_id = {token: i for token, i in enumerate(vocab.keys())}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def tokenize(self, text):
        words = text.split()
        tokens = []
        for word in words:
            word = " ".join(word) + " </w>"
            token = self.bpe_encode(word)
            tokens.extend(token)
        return tokens

    def bpe_encode(self, word):
        if word in self.token_to_id:
            return [word]

        word = word.split()
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                j = word.index(first, i)
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids):
        tokens = [self.id_to_token[id] for id in ids]
        text = " ".join(tokens).replace(" </w>", "").replace(" ", "")
        return text

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.token_to_id, f)

        bpe_ranks_file = os.path.join(save_directory, "bpe_ranks.json")
        if self.bpe_ranks:
            bpe_ranks_serializable = {str(k): v for k, v in self.bpe_ranks.items()}
            with open(bpe_ranks_file, "w") as f:
                json.dump(bpe_ranks_serializable, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, "r") as f:
            token_to_id = json.load(f)

        tokenizer = cls(vocab_size=len(token_to_id))
        tokenizer.token_to_id = token_to_id
        
        bpe_ranks_file = os.path.join(load_directory, "bpe_ranks.json")
        if os.path.exists(bpe_ranks_file):
            with open(bpe_ranks_file, "r") as f:
                bpe_ranks_serializable = json.load(f)
                bpe_ranks = {tuple(map(str.strip, k.strip('()').split(','))): v for k, v in bpe_ranks_serializable.items()}
            tokenizer.bpe_ranks = bpe_ranks
        else:
            tokenizer.bpe_ranks = None
        tokenizer.id_to_token = {i: token for token, i in token_to_id.items()}
        return tokenizer


# scGenT Attention mechanism
class scGenTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_length=MAX_SEQ_LENGTH, dropout=0.1):
        super(scGenTAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        self.max_seq_length = max_seq_length
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length))

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        bsz, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


# scGenT Model Block
class scGenTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, max_seq_length=MAX_SEQ_LENGTH):
        super(scGenTBlock, self).__init__()
        self.norm1_pre = nn.LayerNorm(embed_dim)  # Pre-LayerNorm
        self.attention = scGenTAttention(embed_dim, num_heads, max_seq_length)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1_pre(x)
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


# Full scGenT Model
class scGenTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_length=MAX_SEQ_LENGTH, dropout=0.1):
        super(scGenTModel, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            scGenTBlock(embed_dim, num_heads, ff_hidden_dim, dropout, max_seq_length) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.embed_tokens(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        x = token_embeddings + position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x
        
    def resize_token_embeddings(self, new_vocab_size):
        # Resize the token embeddings to match the new vocabulary size
        old_embeddings = self.embed_tokens
        self.embed_tokens = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
        # Copy over the weights from the old embedding layer to the new one
        self.embed_tokens.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
        self.lm_head = nn.Linear(self.embed_tokens.embedding_dim, new_vocab_size, bias=False)

# Configuration Class for scGenT
class CustomscGenTConfig:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_length, num_labels):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_dict = {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "max_seq_length": self.max_seq_length,
            "num_labels": self.num_labels
        }
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(config_dict, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        config_file = os.path.join(load_directory, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(
            vocab_size=config_dict["vocab_size"],
            n_embd=config_dict["n_embd"],
            n_head=config_dict["n_head"],
            n_layer=config_dict["n_layer"],
            max_seq_length=config_dict["max_seq_length"],
            num_labels=config_dict["num_labels"]
        )


# Custom scGenT model for sequence classification
class CustomscGenTForSequenceClassification(nn.Module):
    def __init__(self, config, class_x=None, class_y=None):
        super(CustomscGenTForSequenceClassification, self).__init__()
        self.transformer = scGenTModel(
            vocab_size=config.vocab_size,
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            num_layers=config.n_layer,
            ff_hidden_dim=4 * config.n_embd,
            max_seq_length=config.max_seq_length
        )
        if class_x is not None and class_y is not None:
            self.classifier = nn.Linear(class_x, class_y)
        else:
            self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.config = config

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        transformer_outputs = self.transformer(input_ids)
        logits = self.classifier(transformer_outputs[:, -1, :])  # Use the last token's hidden state
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

class scGenTModelInitializer:
    def __init__(self, initial_vosize=100, cache_dir=None):
        self.cache_dir = cache_dir
        self.initial_vosize = initial_vosize

    @staticmethod
    def get_tokenizer(vocab_size=100):
        # Initialize the BPETokenizer with the specified vocab size
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        return tokenizer

    @staticmethod
    def get_model_config(n_embd, n_head, n_layer, num_labels, max_seq_length, vocab_size):
        # Create and return a CustomscGenTConfig instance with the specified parameters
        model_config = CustomscGenTConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            max_seq_length=max_seq_length,
            num_labels=num_labels
        )
        return model_config