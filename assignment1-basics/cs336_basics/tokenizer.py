from typing import Self, Iterable, Iterator
import json
import regex as re
from pathlib import Path


class Tokenizer:
    def __init__(
        self: Self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a vocabulary, BPE merge list, and optional special tokens.

        Args:
            vocab: Mapping from token IDs to byte strings that can be decoded into text.
            merges: Ordered list of training merge pairs; each tuple (first, second) represents
                two adjacent byte tokens combined during BPE training.
            special_tokens: Optional list whose tokens get appended to the vocabulary; tokens already
                tracked are skipped and new token IDs are allocated sequentially, so passing None
                leaves the vocabulary unchanged.
        """
        # Initialize vocabulary
        self.vocab_byte = vocab
        self.token_byte_to_id = {v: k for k, v in vocab.items()}

        # Initialize merges
        self.merges_byte = merges
        self.merges_id_to_idx = {}
        for idx in range(len(merges)):
            first, second = merges[idx]
            first_id = self.token_byte_to_id[first]
            second_id = self.token_byte_to_id[second]
            self.merges_id_to_idx[(first_id, second_id)] = idx
        self.max_merge_idx = len(merges)

        self.special_tokens = special_tokens
        if self.special_tokens is None:
            self.special_tokens = []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.all_tokens_bytes = set()
        for _, token in self.vocab_byte.items():
            self.all_tokens_bytes.add(token)

        # Add special tokens to vocabulary
        new_token_id = max(vocab.keys()) + 1
        if self.special_tokens:
            for special_token in self.special_tokens:
                if special_token.encode() not in self.all_tokens_bytes:
                    self.all_tokens_bytes.add(special_token.encode())
                    self.vocab_byte[new_token_id] = special_token.encode()
                    self.token_byte_to_id[special_token.encode()] = new_token_id
                    new_token_id += 1

        self.vocab_size = len(self.vocab_byte)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: Path|str,
        merges_filepath: Path | str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """
        Construct a tokenizer by loading a vocabulary file, BPE merges file and an optional list of special_tokens.

        Args:
            vocab_filepath: Path to the vocabulary file, typically JSON mapping token IDs to byte
                strings (stored as Latin-1) that can be decoded back into bytes.
            merges_filepath: Path to the merges file; each line contains two tokens separated by
                whitespace, representing an adjacent byte pair merged during BPE training.
            special_tokens: Optional list of special tokens to append to the vocabulary; tokens already
                present are skipped, and passing None leaves the vocabulary unchanged.

        Returns:
            Tokenizer: The constructed tokenizer instance.
        """
        with open(vocab_filepath, "r") as f:
            vocab_decoded = json.load(f)
            vocab = {int(k): v.encode("latin-1") for k, v in vocab_decoded.items()}

        with open(merges_filepath, "r") as f:
            merges_decoded = json.load(f)
            merges = [
                (first.encode("latin-1"), second.encode("latin-1"))
                for first, second in merges_decoded
            ]

        return cls(vocab, merges, special_tokens)

    def _merge_pair(
        self,
        pair_to_merge: tuple[int, int],
        word: list[int],
        new_token_id: int,
    ) -> list[int]:
        """Merges occurrences of a pair within a word.

        Args:
            word: The current sequence of token IDs representing the word.
            pair_to_merge: The tuple of (left_id, right_id) tokens to be merged.
            new_token_id: The ID assigned to the new merged token.

        Returns:
            new_word: The updated list of token IDs for the word.
        """
        word_len = len(word)
        new_pair_list: list[int] = []
        left_id, right_id = pair_to_merge

        i = 0
        while i < word_len:
            # Check pair(A, B)
            if (
                (i < word_len - 1)
                and (word[i] == left_id)
                and (word[i + 1] == right_id)
            ):

                # Add new merged token
                new_pair_list.append(new_token_id)
                i += 2  # skip 2 ids
            else:
                new_pair_list.append(word[i])
                i += 1
        return new_pair_list

    def _get_pair_to_merge(self, token_id_list: list[int]) -> tuple[int, int] | None:
        # [(token_id, token_id), (token_id, token_id), ...]
        pair_id_list = list(zip(token_id_list, token_id_list[1:]))

        # [merge1_rank, merge2_rank, ...]
        pair_idx_list = [
            self.merges_id_to_idx.get(pair, self.max_merge_idx + 1)
            for pair in pair_id_list
        ]

        if not pair_idx_list:
            return None
        min_pair_idx = min(pair_idx_list)
        if min_pair_idx == self.max_merge_idx + 1:
            return None
        min_idx = pair_idx_list.index(min_pair_idx)
        return pair_id_list[min_idx]

    def encode(
        self,
        text: str,
    ) -> list[int]:
        # Separate normal text string and special tokens.
        if self.special_tokens:
            text_segments_list = re.split(
                "(" + "|".join(map(re.escape, self.special_tokens)) + ")", text
            )
        else:
            text_segments_list = [text]

        # print(text_segments_list)
        token_ids = []

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat_compiled = re.compile(PAT)

        for segment in text_segments_list:
            if not segment:
                continue
            elif self.special_tokens is not None and segment in self.special_tokens:
                token_ids.append(self.token_byte_to_id[segment.encode()])
            else:  # normal text
                # Pre-tokenize
                for word in re.finditer(pat_compiled, segment):
                    word = word.group(0)
                    word_bytes = word.encode()
                    token_id_list = [
                        self.token_byte_to_id[bytes([byte_val])]
                        for byte_val in word_bytes
                    ]

                    pair_to_merge_id = self._get_pair_to_merge(token_id_list)

                    while pair_to_merge_id is not None:

                        # Get new token id
                        first_id, second_id = pair_to_merge_id
                        first_byte = self.vocab_byte[first_id]
                        second_byte = self.vocab_byte[second_id]
                        # print(first_byte, second_byte)
                        new_token_id = self.token_byte_to_id[first_byte + second_byte]

                        # Merge
                        token_id_list = self._merge_pair(
                            pair_to_merge_id,
                            token_id_list,
                            new_token_id,
                        )

                        pair_to_merge_id = self._get_pair_to_merge(token_id_list)

                    token_ids.extend(token_id_list)

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(
        self,
        ids: list[int],
    ) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: List of token IDs.

        Returns:
            text: Text string.
        """

        # print(f"decoding ids: {ids}")

        text_bytes = bytes()
        for token_id in ids:
            text_bytes += self.vocab_byte[token_id]

        return text_bytes.decode(errors="replace")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "./data/TinyStories_vocab.json",
        "./data/TinyStories_merges.json",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )

    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    encoded_ids = tokenizer.encode(text)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    print("Original string:", text)
    print("Encoded IDs:", encoded_ids)
    print("Tokenized string:", tokenized_string)
