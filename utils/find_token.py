"""
Token index finding for attention map masking.
Adapted from EraseAnything's utils/find_token.py.
"""
import numpy as np


def search_sequence_numpy(arr, seq):
    non_zero_indices = np.where(seq != 0)
    result_indices = []

    for row, col in zip(*non_zero_indices):
        match_indices = np.where(arr[row] == seq[row, col])[0]
        if match_indices.size > 0:
            result_indices.extend([col_idx for col_idx in match_indices])

    return result_indices


def get_word_index(prompt, attn_words, tokenizer_t5, max_length=256):
    """Find token indices of attn_words within the tokenized prompt.

    Padding is skipped in ``search_sequence_numpy`` (``np.where(seq != 0)``).
    T5 still appends EOS after ``attn_words``; the last loop iteration then
    matches EOS in the keyword row to EOS at the end of the prompt, which is
    not a concept token—``[:-1]`` drops that spurious index only.
    """
    prompt_text_ids = tokenizer_t5(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="np",
    ).input_ids

    word_ids = tokenizer_t5(
        attn_words,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="np",
    ).input_ids

    idxs = search_sequence_numpy(prompt_text_ids, word_ids)
    # Drop EOS↔EOS match (see docstring); not needed if attn_words is tokenized with add_special_tokens=False.
    return idxs[:-1]
