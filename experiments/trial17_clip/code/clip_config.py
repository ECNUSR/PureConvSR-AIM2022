''' config '''
trial_name = None

scale = 3

model = dict(
    scale = 3,
    in_channels = 3,
    out_channels = 3,
    channel = 45,
    blocks = 3,
)

data = dict(
    patch_size = 64,
    batch_size = 16,
    iters_per_batch = 1000
)

train = dict(
    lr = 0.0,
    lr_steps = [],
    lr_gamma = 0.5,
    epochs = 20,
    loss = 'mae',
)
