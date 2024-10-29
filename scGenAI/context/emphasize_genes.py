import numpy as np

def emphasize_genes_byfactor(sequences, gene_list, custom_tokenizer, emphasis_factor=2, seed=None):
    reverse_gene_vocab = {v: k for k, v in custom_tokenizer.gene_vocab.items()}
    if seed is not None:
        np.random.seed(seed)
    emphasized_sequences = []
    for genes, expressions in sequences:
        new_genes = list(genes)
        new_expressions = list(expressions)
        emphasize_indices = []
        for gene, expr in zip(genes, expressions):
            if gene in gene_list:
                for _ in range(emphasis_factor - 1):
                    emphasize_indices.append((gene, expr))
        np.random.shuffle(emphasize_indices)
        for gene, expr in emphasize_indices:
            insert_position = np.random.randint(0, len(new_genes) + 1)
            new_genes.insert(insert_position, gene)
            new_expressions.insert(insert_position, expr)
        emphasized_sequences.append((new_genes, new_expressions))
    return emphasized_sequences


