import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRALinear(nn.Module):
    # It's "monkey patch" means you can replace nn.Linear with the new
    # LoRA Linear class without modifying any other code.
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + \
                        self.lora_scale * self.fc_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias

# your implementation

class LoRAConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, rank):
        super().__init__()
        # reduce the dimension to rank and extract features with original kernel size
        self.down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        # restore the dimension to original output dimension
        self.up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)
        # Initialize weights: normal for the down layer and zeros for the up layer
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        # Flatten both weights to two dimensions for matrix multiplication
        print('self.down.weight:', self.down.weight.shape)
        down_weight = self.down.weight.view(self.rank, -1)  # Shape [rank, in_channels * kernel_size_down * kernel_size_down]
        print('down_weight:', down_weight.shape)
        up_weight = self.up.weight.view(self.out_channels, -1)  # Shape [out_channels, rank * 1 * 1]

        # Perform matrix multiplication and then reshape
        # New shape: [out_channels, in_channels, kernel_height, kernel_width]
        combined_weight = torch.matmul(up_weight, down_weight).view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return combined_weight


    @property
    def bias(self):
        return None

class MonkeyPatchLoRAConv2D(nn.Module):
    def __init__(self, conv2d: nn.Conv2d, rank=4, lora_scale=1):
        super().__init__()
        if not isinstance(conv2d, nn.Conv2d):
            raise ValueError(
                f"LoRAConv2DLayer only support nn.Conv2d, but got {type(conv2d )}"
            )
        self.conv = conv2d
        self.rank = rank
        self.lora_scale = lora_scale

        in_channels = conv2d.in_channels
        # print('\nin_channels:/t', in_channels)

        out_channels = conv2d.out_channels
        # print('out_channels:/t', out_channels)
        kernel_size = conv2d.kernel_size
        stride = conv2d.stride
        padding = conv2d.padding
        self.conv2d_lora = LoRAConv2DLayer(in_channels, out_channels, kernel_size, stride, padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states) + \
                        self.lora_scale * self.conv2d_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.conv.weight + self.lora_scale * self.conv2d_lora.weight

    @property
    def bias(self):
        return self.conv.bias

class LoRAConvTranspose2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, rank):
        super().__init__()
        if rank > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_channels, out_channels)}"
            )
        # TODO: down & up order?
        self.down = nn.Conv2d(in_channels, rank, 1, 1, 0, bias=False)
        self.up = nn.ConvTranspose2d(rank, out_channels, kernel_size, stride=stride, padding=padding, output_padding = output_padding, bias=False)

        # Initialize weights: normal for the down layer and zeros for the up layer
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_x = self.down(hidden_states.to(dtype))
        up_x = self.up(down_x)

        return up_x.to(orig_dtype)

    @property
    def weight(self):
        # The down layer reduces dimensionality with a 1x1 kernel, so it's essentially a linear transformation
        # print('\n LoRAConvTranspose2DLayer\n self.down.weight:', self.down.weight.shape)
        down_weight_flat = self.down.weight.view(self.rank, -1)
        # print('down_weight_flat:', down_weight_flat.shape)
        # The up layer performs the transpose convolution
        # To calculate the effective kernel, we need to consider the convolution of each kernel
        # in the up layer with the entire set of down layer weights.
        # Note: This is a conceptual representation and may not directly execute without additional context.
        up_weight_flat = self.up.weight.view(self.out_channels, -1)
        
        combined_weight = torch.matmul(up_weight_flat, down_weight_flat)
        combined_weight = combined_weight.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        return combined_weight


    @property
    def bias(self):
        return None


class MonkeyPatchLoRAConvTranspose2D(nn.Module):
    def __init__(self, convtrans2d: nn.ConvTranspose2d, rank=4, lora_scale=1):
        super().__init__()
        if not isinstance(convtrans2d, nn.ConvTranspose2d):
            raise ValueError(
                f"LoRAConvTranspose2DLayer only support nn.ConvTranspose2d, but got {type(convtrans2d)}"
            )
        self.convtrans = convtrans2d
        self.rank = rank
        self.lora_scale = lora_scale

        in_channels = convtrans2d.in_channels
        out_channels = convtrans2d.out_channels
        kernel_size = convtrans2d.kernel_size
        stride = convtrans2d.stride
        padding = convtrans2d.padding
        output_padding = convtrans2d.output_padding
        self.conv2d_lora = LoRAConvTranspose2DLayer(in_channels, out_channels, kernel_size, stride, padding, output_padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convtrans(hidden_states) + \
                        self.lora_scale * self.conv2d_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.convtrans.weight + self.lora_scale * self.conv2d_lora.weight

    @property
    def bias(self):
        return self.convtrans2d.bias