import torch.nn as nn
from torch.nn import functional as F
import functools


def init_model(model, init_method="normal"):
    """Initialize model parameters.
    Args:
        model: Model to initialize.
        init_method: Name of initialization method: 'normal' or 'xavier'.
    """
    # Initialize model parameters
    if init_method == "normal":
        model.apply(_normal_init)
    elif init_method == "xavier":
        model.apply(_xavier_init)
    else:
        raise NotImplementedError("Invalid weights initializer: {}".format(init_method))


def _normal_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, "weight") and model.weight is not None:
        if class_name.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find("Linear") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def _xavier_init(model):
    """Apply Xavier initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, "weight") and model.weight is not None:
        if class_name.find("Conv") != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find("Linear") != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "group":
        return functools.partial(nn.GroupNorm, num_groups=16)
    else:
        raise NotImplementedError("Invalid normalization type: {}".format(norm_type))


class PatchGAN(nn.Module):
    """PatchGAN discriminator."""

    def __init__(
        self,
        num_channels,
        num_channels_d,
        kernel_size_d,
        initializer="normal",
        norm_type="instance",
        return_binary=False,
        AAAI_type=False,
        **ignore_kwargs
    ):
        """Constructs a basic PatchGAN convolutional discriminator.
        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.
        Args:
            args: Arguments passed in via the command line.
        """
        super(PatchGAN, self).__init__()
        self.return_binary = return_binary
        norm_layer = get_norm_layer(norm_type)

        # Double channels for conditional GAN (concatenated src and tgt images)
        if AAAI_type:
            layers = []
            # layers += [nn.Conv2d(num_channels, num_channels_d, kernel_size_d, stride=2, padding=1),
            #         nn.LeakyReLU(0.2, True)]
            layers += [
                nn.Conv2d(
                    num_channels_d,
                    2 * num_channels_d,
                    kernel_size_d,
                    stride=2,
                    padding=1,
                ),
                norm_layer(2 * num_channels_d),
                nn.LeakyReLU(0.2, True),
            ]
            layers += [
                nn.Conv2d(
                    2 * num_channels_d,
                    4 * num_channels_d,
                    kernel_size_d,
                    stride=2,
                    padding=1,
                ),
                norm_layer(4 * num_channels_d),
                nn.LeakyReLU(0.2, True),
            ]
            # layers += [nn.Conv2d(4 * num_channels_d, 8 * num_channels_d, kernel_size_d, stride=1, padding=1),
            #         norm_layer(8 * num_channels_d),
            #         nn.LeakyReLU(0.2, True)]
            if self.return_binary:
                layers += [
                    nn.Conv2d(4 * num_channels_d, 1, kernel_size_d, stride=1, padding=1)
                ]
            self.model = nn.Sequential(*layers)
        else:
            # self.model=nn.Sequential(
            #     nn.Conv2d(num_channels, num_channels_d, kernel_size_d, stride=1, padding=1),
            #     nn.LeakyReLU(0.2, True),
            #     nn.Conv2d(num_channels_d, 2 * num_channels_d, kernel_size_d, stride=2, padding=1),
            #         norm_layer(2 * num_channels_d),
            #     nn.LeakyReLU(0.2, True),

            #     nn.Conv2d(2 * num_channels_d, 4 * num_channels_d, kernel_size_d, stride=2, padding=1),
            #         norm_layer(4 * num_channels_d),
            #     nn.LeakyReLU(0.2, True),
            #         # nn.Conv2d(4 * num_channels_d, 8 * num_channels_d, kernel_size_d, stride=1, padding=1),
            #         # norm_layer(8 * num_channels_d),
            #         # nn.LeakyReLU(0.2, True),
            # )
            # if self.return_binary:
            #     self.model.append(nn.Conv2d(4 * num_channels_d, 1, kernel_size_d, stride=1, padding=1))
            self.model = nn.Sequential(
                nn.Conv2d(
                    num_channels, num_channels_d, kernel_size_d, stride=2, padding=1
                ),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    num_channels_d,
                    2 * num_channels_d,
                    kernel_size_d,
                    stride=2,
                    padding=1,
                ),
                norm_layer(2 * num_channels_d),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    2 * num_channels_d,
                    4 * num_channels_d,
                    kernel_size_d,
                    stride=2,
                    padding=1,
                ),
                norm_layer(4 * num_channels_d),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    4 * num_channels_d,
                    8 * num_channels_d,
                    kernel_size_d,
                    stride=1,
                    padding=1,
                ),
                norm_layer(8 * num_channels_d),
                nn.LeakyReLU(0.2, True),
            )
            if self.return_binary:
                self.model.append(
                    nn.Conv2d(8 * num_channels_d, 1, kernel_size_d, stride=1, padding=1)
                )
        init_model(self.model, init_method=initializer)

    def forward(self, x):
        out_feat = self.model(x)
        out_feat = F.avg_pool2d(out_feat, out_feat.size()[2:])
        out_feat = out_feat.view(out_feat.size()[0], -1)
        if self.return_binary:
            return out_feat
        else:
            return F.normalize(out_feat, p=2.0, dim=-1)
