from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import backend as K

class LRKaggle(Callback):
    def __init__(self, starting_lr):
        super(LRKaggle, self).__init__()
        self.starting_lr = starting_lr

    def on_train_begin(self, logs = {}):
        self.epoch_counter = 0

    def on_epoch_end(self, epoch, logs = {}):
        self.epoch_counter += 1

        if self.epoch_counter <= 20:
            new_lr = self.starting_lr
        elif self.epoch_counter <= 40:
            new_lr = 0.001
        else:
            new_lr = 0.0001

        K.set_value(self.model.optimizer.lr, new_lr)
        
def checkpoint_factory():
    def factory_f(checkpoint_name):
        return ModelCheckpoint(filepath = checkpoint_name, monitor = 'val_loss', verbose = 1, save_best_only = True)
    return factory_f

def earlystopping_factory():
    def factory_f(patience):
        return EarlyStopping(monitor = 'val_loss', patience = patience, verbose = 0, mode = 'auto')
    return factory_f

def lrkaggle_factory():
    def factory_f(starting_lr):
        return LRKaggle(starting_lr)
    return factory_f

def callback_factory(callback_type):
    if callback_type == 'model_checkpoint':
        return checkpoint_factory()
    elif callback_type == 'early_stopping':
        return earlystopping_factory()
    elif callback_type == 'lr_kaggle':
        return lrkaggle_factory()
    else:
        raise ValueError("Unknown callback type: " + callback_type)