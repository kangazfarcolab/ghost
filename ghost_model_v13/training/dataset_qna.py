import mlx.core as mx
import random

def generate_qa_pairs():
    """
    Generate 100 fixed Q&A pairs for intelligence testing.
    Format: "Q: <question>\nA: <answer>"
    """
    pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What color is the sky?", "Blue"),
        ("What is the boiling point of water?", "100C"),
        ("What is the opposite of fast?", "Slow"),
        ("Is fire hot or cold?", "Hot"),
        ("What language is this?", "English"),
        ("Who is the ghost?", "V12"),
        ("What follows Monday?", "Tuesday"),
        ("Do cats bark?", "No"),
        ("Can birds fly?", "Yes"),
        ("Sun rises in the?", "East"),
        ("Sun sets in the?", "West"),
        ("Red and Blue make?", "Purple"),
        ("Blue and Yellow make?", "Green"),
        ("Ice is solid or liquid?", "Solid"),
        ("Water is solid or liquid?", "Liquid"),
        ("How many legs does a spider have?", "8"),
        ("How many legs does a dog have?", "4"),
    ]
    
    # Expand to 100 using template logic
    for i in range(20):
        pairs.append((f"What number comes after {i}?", f"{i+1}"))
        pairs.append((f"What number comes before {i+1}?", f"{i}"))
        pairs.append((f"Is {i} an even number?", "Yes" if i % 2 == 0 else "No"))
        
    return pairs

class QnADataset:
    def __init__(self, batch_size=32, seq_len=64):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.raw_pairs = generate_qa_pairs()
        self.formatted_data = []
        
        print(f"🧠 Generated {len(self.raw_pairs)} Q&A pairs for benchmark.")
        
        # Tokenize (Byte-level)
        for q, a in self.raw_pairs:
            text = f"Q: {q}\nA: {a}\n"
            tokens = [ord(c) for c in text]
            
            # Pad or Truncate
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens += [0] * (seq_len - len(tokens))
                
            self.formatted_data.append(tokens)
            
        self.data_tensor = mx.array(self.formatted_data, dtype=mx.int32) # [N, L]
        
    def next_batch(self):
        """Returns random batch from the 100 pairs"""
        indices = mx.random.randint(0, len(self.formatted_data), (self.batch_size,))
        batch = self.data_tensor[indices]
        return batch

if __name__ == "__main__":
    ds = QnADataset()
    batch = ds.next_batch()
    print(f"Batch shape: {batch.shape}")
    print("Example sample:")
    text = "".join([chr(t.item()) if t.item() > 0 else "" for t in batch[0]])
    print(f"'{text}'")
