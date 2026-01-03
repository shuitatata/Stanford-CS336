import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
import collections
from functools import partial
import cProfile
import pstats
import tracemalloc
import json


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


def pre_tokenize(
    input_corpus: str,
    special_tokens: list[str],
) -> dict[tuple[int, ...], int]:
    """Pre_tokenize the corpus.

    This function first splits the corpus by the provided special tokens to ensure boundaries are respected. Then, it applies the GPT-2 regex pattern to segment the text into linguistic units (pre-tokens) and encodes them into UTF-8 bytes.

    Args:
        input_corpus: The raw text string (or chunk) to be tokenized.
        special_tokens: A list of string tokens that serve as delimiters. The special tokens must not be separated in the tokenize process.

    Returns:
        counts: A dictionary mapping each unique pre-token (as int tuple) to its frequency count in the corpus.
    """

    # remove special tokens before pre-tokenization
    text_list = re.split("|".join(map(re.escape, special_tokens)), input_corpus)

    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_compiled = re.compile(PAT)

    counts = collections.defaultdict(int)
    for chunk in text_list:
        if chunk:
            for token in re.finditer(pat_compiled, chunk):
                token = token.group(0)
                token_bytes = tuple(token.encode())
                counts[token_bytes] += 1

    return dict(counts)


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
        counts: A dictionary mapping each unique pre-token (as int tuple) to its frequency count this specific chunk.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        decoded_text = f.read(end - start).decode()

    return pre_tokenize(decoded_text, special_tokens)


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
        total_counts = collections.Counter()
        for local_counts in local_counts_list:
            total_counts.update(local_counts)

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

    # Merge pairs
    while len_vocab < vocab_size:
        # Count every pair's frequency
        pair_counts.clear()

        for pre_token, pre_token_freq in total_counts.items():
            pre_token_len = len(pre_token)
            pair_list = [pre_token[i : i + 2] for i in range(0, pre_token_len - 1)]
            for pair in pair_list:
                pair_counts[pair] += pre_token_freq

        # Choose the pair with the highest frequency
        pair_to_merge = max(
            pair_counts.items(),
            key=lambda item: (
                item[1],  # compare frequency first
                vocab[item[0][0]],  # compare BYTE of the first token
                vocab[item[0][1]],  # compare BYTE of the second token
            ),
        )[0]

        # Merge and add to vocabulary
        first, second = pair_to_merge

        # debug
        # print((vocab[first], vocab[second]))
        # if (vocab[first], vocab[second]) == (b" c", b"om"):
        #     print(pair_counts[(first, second)])
        # tops = pair_counts.most_common(10)
        # print_list = []

        # for token, freq in tops:
        #     if vocab[token[0]] == b" c":
        #         stop = True
        #     print_list.append(((vocab[token[0]], vocab[token[1]]), freq))
        # print(print_list)
        # if stop:
        #     input()

        vocab[next_token_id] = vocab[first] + vocab[second]
        len_vocab += 1
        merges.append((vocab[first], vocab[second]))
        new_token_id = next_token_id
        next_token_id += 1

        new_total_counts = collections.Counter()

        for pre_token, pre_token_freq in total_counts.items():
            # Filter the token
            if first not in pre_token:
                new_total_counts[pre_token] = pre_token_freq
                continue

            pre_token_len = len(pre_token)
            new_pair_list = []
            i = 0
            while i < pre_token_len:
                # 检查是否匹配目标 Pair (A, B)
                if (
                    (i < pre_token_len - 1)
                    and (pre_token[i] == first)
                    and (pre_token[i + 1] == second)
                ):

                    new_pair_list.append(new_token_id)
                    i += 2  # 跳过两个
                else:
                    new_pair_list.append(pre_token[i])
                    i += 1

            # 存入新字典 (注意：这里千万不要加 if len >= 2 的判断，防止吞掉单字 token)
            if new_pair_list:
                new_total_counts[tuple(new_pair_list)] += pre_token_freq

        # print((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]))

        # print(vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]])
        # input()

        total_counts = new_total_counts

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
    # main()

    # corpus = "111<|endoftext|>222<|endoftext|>333 444<|endoftext|>"

    # print(pre_tokenize(corpus, ["<|endoftext|>"]))
    pr = cProfile.Profile()
    # tracemalloc.start()
    pr.enable()
    vocab, merges = train_bpe(
        "/Users/shuitata/PlayGround/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        30,
    )

    pr.disable()
    stats = pstats.Stats(pr).sort_stats("cumtime")
    stats.print_stats()

    with open("./data/TinyStories_vocab.json", "w") as f:
        vocab_to_save = {
            str(token_id): token_bytes.decode("latin-1") for token_id, token_bytes in vocab.items()
        }
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
    
    with open("./data/TinyStories_merges.txt", "w") as f:
        for first, second in merges:
            f.write(f"{first.decode("latin-1")} {second.decode("latin-1")}\n")



    # current, peak = tracemalloc.get_traced_memory()

    # print(f"The peak memory use is {peak/1024/1024} MB")
    # tracemalloc.stop()

    # print(vocab)
    # print(merges)
