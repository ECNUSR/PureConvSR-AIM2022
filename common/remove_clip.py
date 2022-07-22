''' remove clip '''
# pylint: disable=no-name-in-module
import numpy as np
import tensorflow as tf


def valid_topological_transformation(model1, model2):
    ''' valid topological transformation '''
    input = (np.random.rand(1, 360, 640, 3) * 255).astype(np.int8)
    input_t = tf.constant(input.clip(0, 255))
    out2 = model2(input_t).numpy()
    out1 = model1(input_t).numpy()

    print(abs(out1 - out2).max())
    print(abs(out1 - out2).sum())
    print(np.allclose(out1, out2))
    assert abs(out1 - out2).max() < 20


def remove_clip(model1, model2):
    ''' main '''
    layers1 = {layer.name: layer for layer in model1.layers}
    layers2 = {layer.name: layer for layer in model2.layers}
    layers1_name = [layer.name for layer in model1.layers]
    else_name = [layer.name for layer in model2.layers if layer.name not in layers1_name]
    for name in layers1_name:
        layer1, layer2 = layers1[name], layers2[name]
        for j in range(len(layer1.weights)):    # pylint: disable=consider-using-enumerate
            layer2.weights[j].assign(layer1.weights[j])
    weight = np.zeros((1, 1, 27, 27))
    for j in range(27):
        weight[0, 0, j, j] = -1
    bias = np.ones(27) * 255
    kernel_min = -np.ones(27)
    kernel_max = np.ones(27)
    for name in else_name:
        layer2 = layers2[name]
        layer2.weights[0].assign(weight)
        layer2.weights[1].assign(bias)
        layer2.weights[3].assign(kernel_min)
        layer2.weights[4].assign(kernel_max)
        layer2.weights[5].assign(0)
        layer2.weights[6].assign(255)
    valid_topological_transformation(model1, model2)
    return model2
