'''
add l2 regularizer on all layers in the model
'''

for layer in my_model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer= regularizers.l2(weight_decay)