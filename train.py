import numpy as np
from minigrad.nn import MLP
from minigrad.tensor import Tensor
from minigrad.optimizers import SGD

def mse_loss(pred, target): return ((pred - target) ** 2).mean()

def train(model, data, targets, epochs=10, learning_rate=0.01):
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(data)
        loss = mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")

def main():
    layers = [512, 256, 128, 64, 10]
    model = MLP(layers, precision=np.float32, weight_init='he', activation='relu')
    batch_size = 5
    data = Tensor(np.random.randn(batch_size, 512), requires_grad=True) # (batch_size, input_size)
    targets = Tensor(np.random.randn(batch_size, 10), requires_grad=False)
    train(model, data, targets, epochs=10, learning_rate=0.01)

if __name__ == "__main__":
    main()
