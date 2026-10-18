import numpy as np
from pathlib import Path


class Perceptron:
    def __init__(self, learning_rate=0.1, memory_file="brain.npz"):
        self.weights = np.zeros(2, dtype=float)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.memory_file = Path(memory_file)

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z > 0 else 0

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            mistakes = 0

            for x, target in zip(X, y):
                prediction = self.predict(x)
                error = target - prediction

                if error != 0:
                    mistakes += 1

                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"mistakes={mistakes}, "
                f"weights={self.weights}, bias={self.bias}"
            )

    def save(self):
        # Save the learned parameters so they survive after the program exits.
        np.savez(
            self.memory_file,
            weights=self.weights,
            bias=np.array(self.bias),
            learning_rate=np.array(self.learning_rate),
        )
        print(f"Memory saved to {self.memory_file}")

    def load(self):
        # Return True if saved parameters were loaded, otherwise start fresh.
        if not self.memory_file.exists():
            print("No saved memory found. Starting fresh.")
            return False

        try:
            with np.load(self.memory_file) as brain:
                weights = brain["weights"]
                bias = float(brain["bias"])

            if weights.shape != self.weights.shape:
                print("Saved memory has the wrong shape. Starting fresh.")
                return False

            self.weights = weights.astype(float)
            self.bias = bias
            print(f"Memory loaded from {self.memory_file}")
            return True
        except (OSError, KeyError, ValueError) as error:
            print(f"Could not load memory ({error}). Starting fresh.")
            return False


if __name__ == "__main__":
    # Training examples for AND: output 1 only when both inputs are 1.
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1])

    neuron = Perceptron(learning_rate=0.1, memory_file="brain.npz")

    # Restore the previous weights and bias if a memory file exists.
    neuron.load()

    # Continue training, then save the updated parameters.
    neuron.train(X, y, epochs=10)
    neuron.save()

    print("\nPredictions:")
    for x in X:
        print(f"{x.astype(int)} -> {neuron.predict(x)}")
