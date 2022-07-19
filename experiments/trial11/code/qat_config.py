''' config '''
trial_name = None

scale = 3

model = dict(
    scale = 3,
)

data = dict(
    patch_size = 64,
    batch_size = 16,
    iters_per_batch = 1000
)

train = dict(
    lr = 1e-3,
    lr_steps = [50, 100, 150, 200],
    lr_gamma = 0.5,
    epochs = 220,
    loss = 'mae',
)
