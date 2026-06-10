import torch
import torch.nn as nn

HIDDEN_SIZE = 16

class MLPIris(nn.Module):
    """
    This is base model without Softmax, because CrossEntropyLoss, which is the loss function that will be used
    when training, will apply Softmax internally.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class MLPIrisInference(nn.Module):
    """
    This is the class for the model on which inference is going to be run on. Since, when training, model will have
    trained weights, we add Softmax here as the classifier during inference.

    Trained model will be passed as a parameter and Softmax is going to be added onto it.
    """

    def __init__(self, base: MLPIris):
        super().__init__()
        self.base = base
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        model_before_softmax = self.base(x)
        model_after_softmax = self.softmax(model_before_softmax)

        return model_after_softmax


def export_onnx(model: MLPIris, path: str) -> None:
    """
    Function to export the model structured in code as an ONNX model.
    When main of this file is run directly, it will call this function and export the untrained model. That is
    necessary so that we can inspect that the graph and the structure is correct.

    Train script can import this function and then, after training, export such trained model.
    """

    inference_model = MLPIrisInference(model)
    inference_model.eval()

    dummy_input = (torch.zeros(1, 4),)      # onnx.export expects inputs to be a tuple

    torch.onnx.export(
        inference_model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )

    print(f"Inference MLP Iris model exported to {path}.")


if __name__ == "__main__":
    model = MLPIris()
    export_onnx(model, "iris_not_trained.onnx")