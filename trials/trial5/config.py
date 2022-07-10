''' config '''
trial_name = None

scale = 3

model = dict(
    scale = 3,
    in_channels = 3,
    out_channels = 3,
    channel = 32,
    blocks = 4,
)

data = dict(
    patch_size = 64,
    batch_size = 16,
    iters_per_batch = 1000
)

train = dict(
    lr = 1e-3,
    lr_steps = [120, 240, 360, 480],
    lr_gamma = 0.5,
    epochs = 600,
    loss = 'mae',
)
