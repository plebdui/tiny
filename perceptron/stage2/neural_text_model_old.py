import numpy as np
from pathlib import Path

TRAINING_TEXT = """
hello there
hello world
help me learn
the cat sat
the cat slept
the dog sat
the dog slept
cats are nice
dogs are nice
we learn by practice
practice makes progress
"""

MEMORY_FILE = Path("neural_brain.npz")


class NeuralTextModel:
    def __init__(self, context_size=8, embedding_size=16,
                 hidden_size=64, learning_rate=0.03, seed=1):
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.chars = []
        self.char_to_id = {}
        self.params = {}

    def _initialize(self):
        vocab_size = len(self.chars)
        self.params["E"] = self.rng.normal(
            0, 0.1, (vocab_size, self.embedding_size)
        )
        input_size = self.context_size * self.embedding_size
        self.params["W1"] = self.rng.normal(
            0, np.sqrt(2 / input_size), (input_size, self.hidden_size)
        )
        self.params["b1"] = np.zeros(self.hidden_size)
        self.params["W2"] = self.rng.normal(
            0, np.sqrt(1 / self.hidden_size), (self.hidden_size, vocab_size)
        )
        self.params["b2"] = np.zeros(vocab_size)

    def _encode_context(self, context):
        context = context[-self.context_size:].rjust(self.context_size)
        return np.array(
            [self.char_to_id.get(ch, self.char_to_id[" "]) for ch in context],
            dtype=np.int64,
        )

    def _forward(self, ids):
        p = self.params
        embedded = p["E"][ids].reshape(-1)
        hidden = np.tanh(embedded @ p["W1"] + p["b1"])
        logits = hidden @ p["W2"] + p["b2"]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return embedded, hidden, probs

    def train(self, text, epochs=100):
        if not self.chars:
            self.chars = sorted(set(text) | {" "})
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self._initialize()

        text = "".join(ch if ch in self.char_to_id else " " for ch in text)
        data = [self.char_to_id[ch] for ch in text]
        examples = [
            (data[i - self.context_size:i], data[i])
            for i in range(self.context_size, len(data))
        ]
        if not examples:
            print("Not enough text to train.")
            return

        p = self.params
        for epoch in range(epochs):
            total_loss = 0.0
            self.rng.shuffle(examples)
            for context_ids, target in examples:
                ids = np.array(context_ids, dtype=np.int64)
                embedded, hidden, probs = self._forward(ids)
                total_loss -= np.log(probs[target] + 1e-12)

                dlogits = probs.copy()
                dlogits[target] -= 1.0
                dW2 = np.outer(hidden, dlogits)
                db2 = dlogits.copy()
                dhidden = p["W2"] @ dlogits
                dz1 = dhidden * (1.0 - hidden**2)
                dW1 = np.outer(embedded, dz1)
                db1 = dz1.copy()
                dembedded = (p["W1"] @ dz1).reshape(
                    self.context_size, self.embedding_size
                )
                dE = np.zeros_like(p["E"])
                for position, char_id in enumerate(ids):
                    dE[char_id] += dembedded[position]

                lr = self.learning_rate
                p["W2"] -= lr * dW2
                p["b2"] -= lr * db2
                p["W1"] -= lr * dW1
                p["b1"] -= lr * db1
                p["E"] -= lr * dE

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: "
                      f"{total_loss / len(examples):.3f}")

    def predict_next(self, context, temperature=0.8):
        if not self.params:
            return "Train or load a model first."
        ids = self._encode_context(context)
        _, _, probs = self._forward(ids)
        temperature = max(float(temperature), 0.05)
        logits = np.log(probs + 1e-12) / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return self.chars[self.rng.choice(len(self.chars), p=probs)]

    def generate(self, prompt, length=120, temperature=0.8):
        result = "".join(
            ch if ch in self.char_to_id else " " for ch in prompt
        )
        for _ in range(length):
            result += self.predict_next(result, temperature)
        return result

    def save(self):
        np.savez(
            MEMORY_FILE,
            chars=np.array(self.chars, dtype="<U1"),
            context_size=np.array(self.context_size),
            embedding_size=np.array(self.embedding_size),
            hidden_size=np.array(self.hidden_size),
            learning_rate=np.array(self.learning_rate),
            **self.params,
        )
        print(f"Neural model saved to {MEMORY_FILE}")

    def load(self):
        if not MEMORY_FILE.exists():
            return False
        with np.load(MEMORY_FILE) as saved:
            self.chars = saved["chars"].tolist()
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self.context_size = int(saved["context_size"])
            self.embedding_size = int(saved["embedding_size"])
            self.hidden_size = int(saved["hidden_size"])
            self.learning_rate = float(saved["learning_rate"])
            self.params = {
                name: saved[name].copy()
                for name in ("E", "W1", "b1", "W2", "b2")
            }
        print(f"Loaded neural model from {MEMORY_FILE}")
        return True


if __name__ == "__main__":
    model = NeuralTextModel()
    if not model.load():
        print("No neural memory found. Starting training.")
        model.train(TRAINING_TEXT, epochs=150)
        model.save()

    print("\nNeural character language model")
    print("Commands: /train, /save, /quit")
    print("Try: hello or the cat")

    while True:
        prompt = input("\nYou: ")
        if prompt == "/quit":
            model.save()
            break
        elif prompt == "/save":
            model.save()
        elif prompt == "/train":
            extra_text = input("Enter additional training text: ")
            filtered = "".join(
                ch if ch in model.char_to_id else " " for ch in extra_text
            )
            model.train(filtered, epochs=30)
            model.save()
        else:
            print("\nModel:\n" + model.generate(prompt))
