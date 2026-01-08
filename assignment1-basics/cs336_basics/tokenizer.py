from typing import Self, Iterable, Iterator


class Tokenizer:
    def __init__(
        self: Self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        pass

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass

    def encode(
        self,
        text: str,
    ) -> list[int]:
        pass

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        pass

    def decode(
        self,
        ids: list[int],
    ) -> str:
        pass
