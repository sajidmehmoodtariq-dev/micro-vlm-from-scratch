import torch
from typing import List, Union

class ByteTokenizer:
    """
    Byte-level tokenizer with exact 260 vocabulary size:
      - 0 to 255: Raw UTF-8 bytes (1-to-1 identity mapping)
      - 256: <PAD> (Used as ignore_index in CrossEntropyLoss)
      - 257: <BOS> (Beginning of sequence)
      - 258: <EOS> (End of sequence / rollout termination)
      - 259: <IMG> (Visual patch delimiter token)
    """
    PAD_TOKEN_ID: int = 256
    BOS_TOKEN_ID: int = 257
    EOS_TOKEN_ID: int = 258
    IMG_TOKEN_ID: int = 259
    VOCAB_SIZE: int = 260

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Converts raw string to UTF-8 byte integers."""
        raw_bytes = list(text.encode("utf-8"))
        tokens = []
        if add_bos:
            tokens.append(self.BOS_TOKEN_ID)
        tokens.extend(raw_bytes)
        if add_eos:
            tokens.append(self.EOS_TOKEN_ID)
        return tokens

    def decode(self, tokens: Union[List[int], torch.Tensor], stop_at_eos: bool = True) -> str:
        """Converts token integers back to a UTF-8 string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()

        byte_list = []
        for t in tokens:
            if stop_at_eos and t == self.EOS_TOKEN_ID:
                break
            if t < 256:
                byte_list.append(t)
        return bytes(byte_list).decode("utf-8", errors="replace")

    def batch_encode(self, texts: List[str], max_len: int = 128) -> torch.Tensor:
        """Pads and batches strings into a 2D LongTensor (batch_size, max_len)."""
        batch = []
        for text in texts:
            tokens = self.encode(text, add_bos=True, add_eos=True)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.PAD_TOKEN_ID] * (max_len - len(tokens))
            batch.append(tokens)
        return torch.tensor(batch, dtype=torch.long)
