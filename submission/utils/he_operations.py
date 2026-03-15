"""
Homomorphic Encryption Operations

Reusable utilities for performing operations on encrypted data (ciphertexts).
"""

def tree_reduce_add(ciphertexts):
    """
    Sum a list of ciphertexts using binary tree reduction.

    Reduces addition depth from O(N) to O(log2 N), which matters for
    bootstrapping budget: each level of the tree is one addition, and
    additions in the same level are independent (parallelisable).

    Args:
        ciphertexts: List of CipherTensor objects to sum (must be non-empty)

    Returns:
        CipherTensor: Sum of all inputs

    Example:
        N=4: [a, b, c, d] -> [(a+b), (c+d)] -> [(a+b+c+d)]   depth 2
        N=4 direct: a -> a+b -> a+b+c -> a+b+c+d              depth 3
    """
    tensors = list(ciphertexts)
    while len(tensors) > 1:
        new_tensors = []
        for i in range(0, len(tensors), 2):
            if i + 1 < len(tensors):
                new_tensors.append(tensors[i] + tensors[i + 1])
            else:
                new_tensors.append(tensors[i])
        tensors = new_tensors
    return tensors[0]


def compute_inner_product_encrypted(ctxt1, ctxt2, embedding_dim=256):
    """
    Compute inner product of two encrypted vectors using HE operations.

    For vectors represented as ciphertexts with elements in slots:
    1. Element-wise multiplication: c = a * b
    2. Sum only the embedding_dim elements using rotation-and-add technique

    The rotation-and-add technique works as follows:
    - Start with vector [a0, a1, a2, a3, a4, a5, a6, a7, ...]
    - Rotate by 1 and add: [a0+a1, a1+a2, a2+a3, a3+a4, ...]
    - Rotate by 2 and add: [a0+a1+a2+a3, a1+a2+a3+a4, ...]
    - Continue until embedding_dim elements are summed
    - First slot contains the sum of the first embedding_dim elements

    Args:
        ctxt1: Encrypted vector (CipherTensor from Orion)
        ctxt2: Encrypted vector (CipherTensor from Orion)
        embedding_dim: Number of elements to sum (default: 256 for CryptoFace)

    Returns:
        CipherTensor: Encrypted scalar containing the inner product in the first slot

    Example:
        >>> # After encrypting two embeddings
        >>> enc_emb1 = orion.encrypt(orion.encode(embedding1, level))
        >>> enc_emb2 = orion.encrypt(orion.encode(embedding2, level))
        >>>
        >>> # Compute encrypted similarity score
        >>> score_ctxt = compute_inner_product_encrypted(enc_emb1, enc_emb2, embedding_dim=256)
        >>>
        >>> # Decrypt only the score (never decrypt embeddings)
        >>> similarity_score = score_ctxt.decrypt().decode()[0]
    """
    # Element-wise multiplication: slot i contains ctxt1[i] * ctxt2[i]
    result = ctxt1 * ctxt2

    # Rotation-and-add butterfly: only sum embedding_dim elements, not all slots.
    # The ciphertext may have 32768 slots but only embedding_dim contain actual data.
    # After log2(embedding_dim) rounds, slot 0 holds the inner product.
    offset = 1
    while offset < embedding_dim:
        rotated = result.roll(offset)
        result = result + rotated
        del rotated
        offset *= 2

    return result
