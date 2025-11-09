import deepxde as dde

def make_deeponet(branch_input_dim, trunk_input_dim, branch_hidden_layers=[128]*4, trunk_hidden_layers= [128]*4, num_outputs=1):
    """
    PyTorch DeepONet (Lu et al., Nature Mach. Intell., 2021)
    Branch: encodes input function u(theta, phi)
    Trunk:  encodes output coordinates (r, theta, phi)
    """
    branch_layers = [branch_input_dim] + branch_hidden_layers
    trunk_layers  = [trunk_input_dim]  + trunk_hidden_layers

    net = dde.nn.DeepONet(
        layer_sizes_branch=branch_layers,
        layer_sizes_trunk=trunk_layers,
        activation="tanh",
        kernel_initializer="Glorot normal",
        num_outputs=num_outputs,   # e.g., 1 if predicting one scalar velocity value
    )
    return net