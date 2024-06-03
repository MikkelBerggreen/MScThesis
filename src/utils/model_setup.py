import torch
import models.models as models
import metrics.lossfunctions as lossfunctions

# This file contains helper functions for getting specific objects based on names provided in configs

def get_model_from_name(model_name, input_shape, output_shape, dropout=None, avgpool_size=None, n_blocks=None):
    # Assuming all models are in the models.models module as per your import statement.
    # First, we try to get the class reference from the module based on the model_name string.
    try:
        ModelClass = getattr(models, model_name)
    except AttributeError:
        raise NotImplementedError(f'Model {model_name} not implemented')
    
    # Check if dropout is needed based on the class constructor.
    if 'dropout' in ModelClass.__init__.__code__.co_varnames:
        if 'n_blocks' in ModelClass.__init__.__code__.co_varnames:
            model = ModelClass(input_shape, output_shape, dropout, n_blocks)
        else:
            model = ModelClass(input_shape, output_shape, dropout)
    elif 'avgpool_size' in ModelClass.__init__.__code__.co_varnames:
        model = ModelClass(input_shape, output_shape, avgpool_size)
    else:
        model = ModelClass(input_shape, output_shape)

    return model

def get_loss_function_from_name(loss_function_name, task):
    # Built-in PyTorch losses are directly accessible via torch.nn
    builtin_losses = {
        'MSELoss': torch.nn.MSELoss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss
    }

    # For EEG task, we may use both custom and built-in loss functions
    if task == 'eeg':
        if loss_function_name in builtin_losses:
            criterion = builtin_losses[loss_function_name]()
        else:
            try:
                # Attempt to get custom loss function
                LossClass = getattr(lossfunctions, loss_function_name)
                criterion = LossClass()
            except AttributeError:
                raise NotImplementedError(f'Loss function {loss_function_name} not implemented')
    # For classification task, currently only using CrossEntropyLoss
    elif task == 'class':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown task {task}')

    return criterion

def get_optimizer_from_name(optimizer_name, model, lr, weight_decay):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} not implemented')

    return optimizer