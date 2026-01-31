import os
from typing import BinaryIO, Self
from multiprocessing import Pool
import regex as re
import collections
from functools import partial
import cProfile
import pstats
import tracemalloc
import json
import heapq
import time


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class PairItem:
    def __init__(self, pair, first_bytes, second_bytes, freq):
        self.pair = pair
        self.freq = freq
        self.first_bytes = first_bytes
        self.second_bytes = second_bytes

    def __lt__(self, other: Self):
        if self.freq > other.freq:
            return True
        elif self.freq < other.freq:
            return False
        else:
            return (self.first_bytes, self.second_bytes) > (
                other.first_bytes,
                other.second_bytes,
            )

    def __repr__(self):
        return f"({self.pair}, {self.first_bytes}, {self.second_bytes}, {self.freq})"


def pre_tokenize(
    input_corpus: str,
    special_tokens: list[str],
) -> collections.Counter:
    """Pre_tokenize the corpus.

    This function first splits the corpus by the provided special tokens to ensure boundaries are respected. Then, it applies the GPT-2 regex pattern to segment the text into linguistic units (pre-tokens) and encodes them into UTF-8 bytes.

    Args:
        input_corpus: The raw text string (or chunk) to be tokenized.
        special_tokens: A list of string tokens that serve as delimiters. The special tokens must not be separated in the tokenize process.

    Returns:
        counts: A collections.Counter mapping each unique pre-token (as int tuple) to its frequency count in the corpus.
    """

    # remove special tokens before pre-tokenization
    text_list = re.split("|".join(map(re.escape, special_tokens)), input_corpus)

    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_compiled = re.compile(PAT)

    counts = collections.Counter()
    for chunk in text_list:
        if chunk:
            for token in re.finditer(pat_compiled, chunk):
                token = token.group(0)
                token_bytes = tuple(token.encode())
                counts[token_bytes] += 1

    return counts


def _bpe_worker(
    input_path: str,
    special_tokens: list[str],
    start: int,
    end: int,
) -> dict[tuple[int, ...], int]:
    """Reads a specific chunk of the input file and computes pre-token frequencies.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        special_tokens: A list of string tokens that must not be separated in the tokenize process.
        start: The start byte offset to begin reading from.
        end: The end byte offset to stop reading at.

    Returns:
        counts: A collections.Counter mapping each unique pre-token (as int tuple) to its frequency count this specific chunk.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        decoded_text = f.read(end - start).decode()

    return pre_tokenize(decoded_text, special_tokens)


def _merge_pair(
    pair_to_merge: tuple[int, int],
    word: list[int],
    new_token_id: int,
    word_freq: int,
) -> tuple[tuple[int], dict[tuple[int, int], int]]:
    """Merges occurrences of a pair within a word and calculates frequency updates.

    Args:
        word: The current sequence of token IDs representing the word.
        pair_to_merge: The tuple of (left_id, right_id) tokens to be merged.
        new_token_id: The ID assigned to the new merged token.
        word_freq: The frequency count of this word in the corpus.

    Returns:
        new_word: The updated list of token IDs for the word.
        changes: A dictionary mapping token pairs to their frequency change
            (e.g., {(prev, left): -freq, (prev, new): +freq}).
    """
    word_len = len(word)
    new_pair_list = []
    changes = collections.defaultdict(int)
    left_id, right_id = pair_to_merge
    i = 0
    while i < word_len:
        # Check pair(A, B)
        if (i < word_len - 1) and (word[i] == left_id) and (word[i + 1] == right_id):
            if i > 0:
                prev_token = word[i - 1]
                changes[(prev_token, left_id)] -= word_freq
                changes[(prev_token, new_token_id)] += word_freq

            if i + 2 < word_len:
                next_token = word[i + 2]
                changes[(right_id, next_token)] -= word_freq
                changes[(new_token_id, next_token)] += word_freq

            # Add new merged token
            new_pair_list.append(new_token_id)
            i += 2  # skip 2 bytes
        else:
            new_pair_list.append(word[i])
            i += 1
    return tuple(new_pair_list), changes


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains the BPE tokenizer

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
        num_processes: Number of parallel processes to use for training.

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

        merges: A list of BPE merges produced from traning. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. the merges should be ordered by order of creation.
    """
    with open(input_path, "rb") as f:
        first_special_token_bytes = special_tokens[0].encode()
        boundaries = find_chunk_boundaries(f, num_processes, first_special_token_bytes)

    # Paralled pre tokenize
    with Pool(num_processes) as p:
        local_counts_list = p.starmap(
            partial(_bpe_worker, input_path, special_tokens),
            zip(boundaries[:-1], boundaries[1:]),
        )
        word_counts: collections.Counter = collections.Counter()
        for local_counts in local_counts_list:
            word_counts.update(local_counts)

    # Initialize vocab with standard ASCII/Byte range
    vocab = {k: bytes([k]) for k in range(256)}
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode()
        next_token_id += 1
    pair_counts = collections.Counter()

    # Initialize merges list
    merges = list()
    len_vocab = len(vocab)

    # Initialize pair_to_word map
    pair_to_word = collections.defaultdict(set)

    # Calculate inital pair frequency
    for word, word_freq in word_counts.items():
        word_len = len(word)
        pair_list = zip(word, word[1:])
        for pair in pair_list:
            pair_counts[pair] += word_freq
            pair_to_word[pair].add(word)

    # Initialize pair frequncy heapq
    pair_counts_heap: list[PairItem] = []
    for pair, pair_freq in pair_counts.items():
        pair_item = PairItem(pair, vocab[pair[0]], vocab[pair[1]], pair_freq)
        pair_counts_heap.append(pair_item)
    heapq.heapify(pair_counts_heap)

    # Merge pairs
    while len_vocab < vocab_size:
        # Choose the pair with the highest frequency
        max_pair_item = heapq.heappop(pair_counts_heap)
        while max_pair_item.freq != pair_counts[max_pair_item.pair]:
            max_pair_item = heapq.heappop(pair_counts_heap)

        # Merge and add to vocabulary
        max_pair = max_pair_item.pair
        left_id, right_id = max_pair

        vocab[next_token_id] = vocab[left_id] + vocab[right_id]
        len_vocab += 1
        merges.append((vocab[left_id], vocab[right_id]))
        new_token_id = next_token_id
        next_token_id += 1

        pairs_to_update = set()
        affected_words = pair_to_word.pop(max_pair, set())

        for word in affected_words:
            if word not in word_counts:
                continue

            word_freq = word_counts[word]
            word_counts.pop(word)

            new_word, changes = _merge_pair(max_pair, word, new_token_id, word_freq)
            for pair in zip(new_word, new_word[1:]):
                pair_to_word[pair].add(new_word)

            for pair, delta in changes.items():
                pair_counts[pair] += delta
                pairs_to_update.add(pair)

            # Save to new dict
            if new_word:
                word_counts[new_word] += word_freq

        # Update the heap
        for pair in pairs_to_update:
            pair_item = PairItem(
                pair, vocab[pair[0]], vocab[pair[1]], pair_counts[pair]
            )
            heapq.heappush(pair_counts_heap, pair_item)

        # Delete merged pair from dict
        pair_counts.pop(max_pair_item.pair)

    return vocab, merges


def main():
    ## Usage
    with open("./data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        print(boundaries[0])

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(start, end)
            f.seek(start)
            print(f.read(end - start))

            break


if __name__ == "__main__":

    start_time = time.time()

    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        40,
    )

    mid_time = time.time()
    print(f"Training time: {mid_time - start_time} seconds")

    with open("./data/TinyStories_vocab.json", "w") as f:
        vocab_to_save = {
            str(token_id): token_bytes.decode("latin-1")
            for token_id, token_bytes in vocab.items()
        }
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

    with open("./data/TinyStories_merges.json", "w") as f:
        merges_to_save = [
            (first.decode("latin-1"), second.decode("latin-1"))
            for first, second in merges
        ]
        json.dump(merges_to_save, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Saving time: {end_time - mid_time} seconds")

    # current, peak = tracemalloc.get_traced_memory()

    # print(f"The peak memory use is {peak/1024/1024} MB")
    # tracemalloc.stop()

    # At index 31 diff: (b' ', b'd') != (b' a', b'nd')
