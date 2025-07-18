import numpy as np
from minigrad.nn import MLP, mse_loss, SGD
from minigrad.tensor import Tensor

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
    print(model)
    print(model.parameters())
    data = Tensor(np.random.randn(1, 512), requires_grad=True)
    targets = Tensor(np.random.randn(1, 512), requires_grad=False)
    train(model, data, targets, epochs=10, learning_rate=0.01)

if __name__ == "__main__":
    main()
