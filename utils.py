##### HELPER FUNCTIONS #####
from torchprofile import profile_macs

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs



def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

