import numpy as np
from pathlib import Path

TRAINING_FILE = Path("training.txt")
MODEL_FILE = Path("tiny_transformer.npz")

DEFAULT_TEXT = """
hello there
hello world
the cat sat
the cat slept
the dog sat
the dog slept
cats are nice
dogs are nice
we learn by practice
practice makes progress
A neural network learns patterns from examples.
A language model predicts the next character based on previous characters.
Python can train a neural network using numerical calculations.
Training on more varied text helps a model learn more varied patterns.
A model improves by comparing its predictions with the correct answers.
The network adjusts its weights to reduce prediction errors.
"""


class TinyTransformer:
    """
    A deliberately small character-level transformer implemented with NumPy.
    One causal self-attention head, one tanh feed-forward-style output block,
    and manual backpropagation. Educational, not production-scale.
    """

    def __init__(self, block_size=16, d_model=24, seed=7, learning_rate=0.01):
        self.block_size = block_size
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.chars = []
        self.char_to_id = {}
        self.p = {}

    def _init_params(self):
        v, d = len(self.chars), self.d_model
        scale = 1.0 / np.sqrt(d)
        self.p = {
            "E": self.rng.normal(0, 0.08, (v, d)),
            "P": self.rng.normal(0, 0.04, (self.block_size, d)),
            "Wq": self.rng.normal(0, scale, (d, d)),
            "Wk": self.rng.normal(0, scale, (d, d)),
            "Wv": self.rng.normal(0, scale, (d, d)),
            "Wo": self.rng.normal(0, scale, (d, d)),
            "Wout": self.rng.normal(0, scale, (d, v)),
            "bout": np.zeros(v),
        }

    def _expand_vocab(self, text):
        new_chars = sorted(set(self.chars) | set(text) | {" "})
        if not self.p:
            self.chars = new_chars
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self._init_params()
            return
        if new_chars == self.chars:
            return

        old_chars = self.chars[:]
        old_p = self.p
        old_map = {ch: i for i, ch in enumerate(old_chars)}
        new_map = {ch: i for i, ch in enumerate(new_chars)}
        new_v = len(new_chars)
        E = self.rng.normal(0, 0.08, (new_v, self.d_model))
        Wout = self.rng.normal(
            0, 1.0 / np.sqrt(self.d_model), (self.d_model, new_v)
        )
        bout = np.zeros(new_v)
        for ch, old_i in old_map.items():
            j = new_map[ch]
            E[j] = old_p["E"][old_i]
            Wout[:, j] = old_p["Wout"][:, old_i]
            bout[j] = old_p["bout"][old_i]
        self.chars = new_chars
        self.char_to_id = new_map
        self.p = dict(old_p)
        self.p["E"], self.p["Wout"], self.p["bout"] = E, Wout, bout

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def _forward(self, ids):
        p = self.p
        length = len(ids)
        x = p["E"][ids] + p["P"][:length]
        q = x @ p["Wq"]
        k = x @ p["Wk"]
        v = x @ p["Wv"]

        scores = (q @ k.T) / np.sqrt(self.d_model)
        mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        att = self._softmax(scores, axis=1)
        values = att @ v
        h = np.tanh(x + values @ p["Wo"])
        logits = h @ p["Wout"] + p["bout"]
        probs = self._softmax(logits, axis=1)
        cache = (ids, x, q, k, v, att, values, h, probs, mask)
        return probs, cache

    def _train_one(self, ids, targets):
        p = self.p
        probs, cache = self._forward(ids)
        (ids, x, q, k, v, att, values, h, probs, mask) = cache
        length = len(ids)
        loss = -np.mean(np.log(probs[np.arange(length), targets] + 1e-12))

        dlogits = probs.copy()
        dlogits[np.arange(length), targets] -= 1
        dlogits /= length

        grads = {}
        grads["Wout"] = h.T @ dlogits
        grads["bout"] = np.sum(dlogits, axis=0)
        dh = dlogits @ p["Wout"].T
        dz = dh * (1 - h * h)

        grads["Wo"] = values.T @ dz
        dvalues = dz @ p["Wo"].T
        dx = dz.copy()  # residual path

        datt = dvalues @ v.T
        dv = att.T @ dvalues
        dscores = att * (datt - np.sum(datt * att, axis=1, keepdims=True))
        dscores[mask] = 0.0

        scale = np.sqrt(self.d_model)
        dq = dscores @ k / scale
        dk = dscores.T @ q / scale

        grads["Wq"] = x.T @ dq
        grads["Wk"] = x.T @ dk
        grads["Wv"] = x.T @ dv
        dx += dq @ p["Wq"].T
        dx += dk @ p["Wk"].T
        dx += dv @ p["Wv"].T

        grads["P"] = np.zeros_like(p["P"])
        grads["P"][:length] = dx
        grads["E"] = np.zeros_like(p["E"])
        for pos, char_id in enumerate(ids):
            grads["E"][char_id] += dx[pos]

        # Clip the global gradient magnitude to reduce unstable updates.
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        factor = min(1.0, 1.0 / (norm + 1e-12))
        for name, grad in grads.items():
            p[name] -= self.learning_rate * factor * grad
        return float(loss)

    def train(self, text, epochs=100):
        text = text.strip("\ufeff")
        if len(text) <= self.block_size:
            print(f"Need more than {self.block_size} characters of training text.")
            return
        self._expand_vocab(text)
        data = np.array([self.char_to_id[ch] for ch in text], dtype=np.int64)
        starts = list(range(0, len(data) - self.block_size, 3))
        if not starts:
            starts = [0]

        for epoch in range(epochs):
            self.rng.shuffle(starts)
            losses = []
            for start in starts:
                seq = data[start:start + self.block_size + 1]
                ids, targets = seq[:-1], seq[1:]
                losses.append(self._train_one(ids, targets))
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {np.mean(losses):.3f}")

    def generate(self, prompt, length=250, temperature=0.8):
        if not self.p:
            return "No model loaded. Run /train first."
        result = "".join(ch if ch in self.char_to_id else " " for ch in prompt)
        for _ in range(length):
            context = result[-self.block_size:]
            ids = np.array(
                [self.char_to_id.get(ch, self.char_to_id[" "]) for ch in context],
                dtype=np.int64
            )
            probs, _ = self._forward(ids)
            next_probs = probs[-1].copy()
            temp = max(float(temperature), 0.05)
            logits = np.log(next_probs + 1e-12) / temp
            next_probs = self._softmax(logits)
            next_id = self.rng.choice(len(self.chars), p=next_probs)
            result += self.chars[next_id]
        return result

    def save(self):
        if not self.p:
            print("Nothing to save.")
            return
        np.savez(
            MODEL_FILE,
            chars=np.array(self.chars, dtype="<U1"),
            block_size=np.array(self.block_size),
            d_model=np.array(self.d_model),
            learning_rate=np.array(self.learning_rate),
            **self.p
        )
        print(f"Transformer saved to {MODEL_FILE}")

    def load(self):
        if not MODEL_FILE.exists():
            return False
        with np.load(MODEL_FILE) as data:
            self.chars = data["chars"].tolist()
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self.block_size = int(data["block_size"])
            self.d_model = int(data["d_model"])
            self.learning_rate = float(data["learning_rate"])
            names = ("E", "P", "Wq", "Wk", "Wv", "Wo", "Wout", "bout")
            self.p = {name: data[name].copy() for name in names}
        print(f"Loaded transformer from {MODEL_FILE}")
        return True


def load_text():
    if TRAINING_FILE.exists():
        text = TRAINING_FILE.read_text(encoding="utf-8")
        print(f"Read {len(text)} characters from {TRAINING_FILE}.")
        return text
    print(f"{TRAINING_FILE} not found; using built-in starter text.")
    return DEFAULT_TEXT


if __name__ == "__main__":
    model = TinyTransformer()
    model.load()
    print("\nTiny NumPy Transformer")
    print("Commands: /train, /add, /save, /quit")
    print("This is a small educational model; generation may be rough.")

    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            model.save()
            break
        command = prompt.strip().lower()
        if command == "/quit":
            model.save()
            break
        elif command == "/save":
            model.save()
        elif command == "/train":
            model.train(load_text(), epochs=100)
            model.save()
        elif command == "/add":
            addition = input("Text to append to training.txt: ")
            if addition.strip():
                with TRAINING_FILE.open("a", encoding="utf-8") as f:
                    f.write("\n" + addition.strip() + "\n")
                print("Added text. Run /train to train on the updated file.")
        elif command:
            print("\nModel:\n" + model.generate(prompt))
