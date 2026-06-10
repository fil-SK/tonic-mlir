import torch
import torch.nn as nn

HIDDEN_SIZE = 128


class MLPMNIST(nn.Module):
    """
    Similarly to MLP Iris.
    Model to be used for training.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class MLPMNISTInference(nn.Module):
    """
    Similarly to MLP Iris.
    Model to be used for inference.
    """

    def __init__(self, base: MLPMNIST):
        super().__init__()
        self.base = base
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output_no_softmax = self.base(x)
        output_softmax = self.softmax(output_no_softmax)

        return output_softmax


def export_onnx(model: MLPMNIST, path: str) -> None:
    inference_model = MLPMNISTInference(model)
    inference_model.eval()

    dummy_input = (torch.zeros(1, 1, 28, 28), )

    torch.onnx.export(
        inference_model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )

    print(f"Inference MLP MNIST model exported to {path}.")


if __name__ == "__main__":
    model = MLPMNIST()
    export_onnx(model, "mlp_mnist_not_trained.onnx")