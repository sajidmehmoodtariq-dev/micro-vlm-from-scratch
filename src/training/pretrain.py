import os
import sys
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# Robust path resolution to allow running directly or as module
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)
_ROOT_DIR = os.path.dirname(_SRC_DIR)

for path in [_ROOT_DIR, _SRC_DIR, os.path.join(_SRC_DIR, "model")]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from model import ByteTokenizer, MicroMultimodalMoE
except ImportError:
    from src.model import ByteTokenizer, MicroMultimodalMoE

# -------------------------------------------------------------
# 1. Synthetic Multimodal Reasoning Dataset
# -------------------------------------------------------------
class SyntheticReasoningDataset(Dataset):
    """
    Generates algorithmic logic samples and synthetic visual-color associations.
    Example text:
      "Q: True and (False or not False)? A: True"
      "Q: x = 4; x += 3; print(x) A: 7"
      "IMG: [visual shape] Q: What color is the circle? A: red"
    """
    def __init__(self, tokenizer: ByteTokenizer, size: int = 5000, max_len: int = 128):
        self.tokenizer = tokenizer
        self.size = size
        self.max_len = max_len
        self.colors = ["red", "green", "blue", "yellow", "white"]

    def __len__(self):
        return self.size

    def _generate_boolean_logic(self) -> str:
        ops = ["and", "or"]
        vals = ["True", "False"]
        a, b = random.choice(vals), random.choice(vals)
        op = random.choice(ops)
        res = eval(f"{a} {op} {b}")
        return f"Logic: {a} {op} {b} -> {res}"

    def _generate_arithmetic(self) -> str:
        a = random.randint(0, 50)
        b = random.randint(1, 50)
        op = random.choice(["+", "-", "*"])
        res = eval(f"{a} {op} {b}")
        return f"Calc: {a} {op} {b} = {res}"

    def _generate_multimodal_sample(self) -> Tuple[str, torch.Tensor]:
        """Creates a synthetic 64x64 color patch with a grounded text description."""
        color_idx = random.randint(0, len(self.colors) - 1)
        chosen_color = self.colors[color_idx]

        # 64x64 canvas
        img = torch.zeros(3, 64, 64, dtype=torch.float32)
        if chosen_color == "red":
            img[0, :, :] = 1.0
        elif chosen_color == "green":
            img[1, :, :] = 1.0
        elif chosen_color == "blue":
            img[2, :, :] = 1.0
        elif chosen_color == "yellow":
            img[0, :, :] = 1.0
            img[1, :, :] = 1.0
        elif chosen_color == "white":
            img[:, :, :] = 1.0

        # Exactly 16 <IMG> tokens matching downsampler: (64/8)^2 / 4 = 16
        img_tokens = "<IMG>" * 16
        text = f"Visual {img_tokens} Question: Primary tint? Answer: {chosen_color}."
        return text, img

    def __getitem__(self, idx: int):
        sample_type = random.random()
        pixel_values = torch.zeros(3, 64, 64, dtype=torch.float32)

        if sample_type < 0.4:
            raw_text = self._generate_boolean_logic()
        elif sample_type < 0.7:
            raw_text = self._generate_arithmetic()
        else:
            raw_text, pixel_values = self._generate_multimodal_sample()

        # Tokenize
        tokens = []
        tokens.append(self.tokenizer.BOS_TOKEN_ID)
        
        # Tokenize text while properly embedding IMG_TOKEN_ID
        parts = raw_text.split("<IMG>")
        for i, part in enumerate(parts):
            if part:
                tokens.extend(list(part.encode("utf-8")))
            if i < len(parts) - 1:
                tokens.append(self.tokenizer.IMG_TOKEN_ID)
                
        tokens.append(self.tokenizer.EOS_TOKEN_ID)

        # Padding / Truncating
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.PAD_TOKEN_ID] * (self.max_len - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values
        }

# -------------------------------------------------------------
# 2. Pre-training Loop
# -------------------------------------------------------------
def train_pretrain(epochs: int = 5, save_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if save_dir is None:
        save_dir = os.path.join(_ROOT_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pretrained_25m.pt")

    tokenizer = ByteTokenizer()
    model = MicroMultimodalMoE(
        vocab_size=260,
        d_model=256,
        num_layers=8,
        num_heads=8,
        num_experts=4,
        top_k=2,
        window_size=32,
        max_seq_len=128
    ).to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)

    dataset = SyntheticReasoningDataset(tokenizer=tokenizer, size=2000, max_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    print(f"\nStarting Pre-training ({epochs} Epochs)...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0.0
        total_lm_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, pixel_values=pixel_values, labels=labels)

            loss = outputs["loss"]
            loss.backward()

            # Gradient clipping to prevent MoE router gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            total_lm_loss += outputs["lm_loss"].item()

            if (step + 1) % 25 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(dataloader)}] | "
                    f"LM Loss: {outputs['lm_loss'].item():.4f} | "
                    f"Aux MoE Loss: {outputs['aux_loss'].item():.4f}"
                )

        avg_lm = total_lm_loss / len(dataloader)
        print(f"--- Epoch {epoch} Complete | Average LM Loss: {avg_lm:.4f} ---")

    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\nPretrained checkpoint saved successfully to {save_path}")

if __name__ == "__main__":
    print("Initializing Pre-training Script...")
    train_pretrain()
