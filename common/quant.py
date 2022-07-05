''' quant '''
import tensorflow_model_optimization as tfmot


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    ''' NoOpQuantizeConfig '''
    # pylint: disable=arguments-renamed
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_anctivations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}


def ps_quantization(layer):
    ''' ps_quantization '''
    if 'lambda' in layer.name or 'no_quant' in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    return layer
