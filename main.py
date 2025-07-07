import torch.nn.utils.parametrize as parametrize
from base_model import baseNet
from LoRA_layer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = baseNet().to(device)


parametrize.register_parametrization(
    model.linear1.linear, "weight", linear_layer_parameterization(model.linear1.linear, device)
)
parametrize.register_parametrization(
    model.linear2.linear, "weight", linear_layer_parameterization(model.linear2.linear, device)
)
parametrize.register_parametrization(
    model.linear3.linear, "weight", linear_layer_parameterization(model.linear3.linear, device)
)


def enable_disable_lora(enabled=True):
    for layer in [model.linear1.linear, model.linear2.linear, model.linear3.linear]:
        layer.parametrizations["weight"][0].enabled = enabled

if __name__ == "__main__":
    input = torch.randn((4, 100), device=device)
    enable_disable_lora(enabled=True)
    print(model(input).shape)
    enable_disable_lora(enabled=False)
    print(model(input).shape)