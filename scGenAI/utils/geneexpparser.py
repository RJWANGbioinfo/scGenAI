import os

def create_reverse_expression_vocab(custom_tokenizer):
    return {v: k for k, v in custom_tokenizer.expression_vocab.items()}

def create_reverse_gene_vocab(custom_tokenizer):
    return {v: k for k, v in custom_tokenizer.gene_vocab.items()}

class BiDirectionalDict:
    def __init__(self, mapping):
        self.name_to_id = mapping  # Gene name to token ID
        self.id_to_name = {v: k for k, v in mapping.items()}  # Token ID to gene name

    def get_token_id(self, gene_name):
        return self.name_to_id.get(gene_name)

    def get_gene_name(self, token_id):
        return self.id_to_name.get(token_id)

def read_gmt_to_dict(file_path):
    gmt_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_set = parts[0]
            genes = parts[2:]
            gmt_dict[gene_set] = genes
    return gmt_dict
    
