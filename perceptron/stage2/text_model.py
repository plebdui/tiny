
import numpy as np
from pathlib import Path

TRAINING_TEXT = """
hello
hello
hello
help
help
hey
he
hello there
the cat sat
the cat slept
the dog sat
the dog slept
cats are nice
dogs are nice
"""

MEMORY_FILE = Path("text_brain_v2.npz")


class TextModel:
    def __init__(self, context_size=4):
        self.context_size = context_size
        self.transitions = {}
        self.rng = np.random.default_rng()

    def train(self, text):
        # Learn which character can follow each character sequence.
        for i in range(len(text) - self.context_size):
            context = text[i:i + self.context_size]
            next_char = text[i + self.context_size]

            if context not in self.transitions:
                self.transitions[context] = {}

            self.transitions[context][next_char] = (
                self.transitions[context].get(next_char, 0) + 1
            )

    def predict_next(self, context):
        # Try the longest available context first.
        for size in range(min(self.context_size, len(context)), 0, -1):
            key = context[-size:]

            # Find transitions beginning with this context.
            matches = {}
            for learned_context, options in self.transitions.items():
                if learned_context.endswith(key):
                    for char, count in options.items():
                        matches[char] = matches.get(char, 0) + count

            if matches:
                chars = list(matches.keys())
                counts = np.array(
                    [matches[c] for c in chars], dtype=float
                )
                probabilities = counts / counts.sum()
                return self.rng.choice(
                    chars, p=probabilities
                )

        # If nothing matches, start with a newline.
        return "\n"

    def generate(self, prompt, length=100):
        result = prompt

        for _ in range(length):
            result += self.predict_next(result)

        return result

    def save(self):
        # Store the learned transitions in a portable format.
        contexts = []
        next_chars = []
        counts = []

        for context, options in self.transitions.items():
            for char, count in options.items():
                contexts.append(context)
                next_chars.append(char)
                counts.append(count)

        np.savez(
            MEMORY_FILE,
            contexts=np.array(contexts, dtype=str),
            next_chars=np.array(next_chars, dtype=str),
            counts=np.array(counts, dtype=np.int64),
            context_size=np.array(self.context_size),
        )
        print("Memory saved.")

    def load(self):
        if not MEMORY_FILE.exists():
            return False

        with np.load(MEMORY_FILE) as brain:
            self.context_size = int(brain["context_size"])

            for context, char, count in zip(
                brain["contexts"],
                brain["next_chars"],
                brain["counts"],
            ):
                context = str(context)
                char = str(char)

                if context not in self.transitions:
                    self.transitions[context] = {}

                self.transitions[context][char] = int(count)

        print("Memory loaded.")
        return True


if __name__ == "__main__":
    model = TextModel(context_size=4)

    if not model.load():
        model.train(TRAINING_TEXT)
        model.save()

    print("\nCharacter-level text model")
    print("Commands: /train, /quit")

    while True:
        prompt = input("\nYou: ")

        if prompt == "/quit":
            break

        elif prompt == "/train":
            new_text = input("Enter training text: ")
            model.train(new_text)
            model.save()
            print("Learned the new text.")

        else:
            print("\nModel:", model.generate(prompt, length=100))
