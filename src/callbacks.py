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

class HistoryWriter(Callback):
    def __init__(self, filepath):
        super(HistoryWriter, self).__init__()
        self.filepath = filepath
    
    def on_train_begin(self, logs = {}):
        self.epoch_counter = 0
        with open(self.filepath, "a+") as f:
            f.write(",".join(["epoch", "tr_loss", 'tr_acc', 'val_loss', 'val_acc']))
            f.write('\n')
    
    def on_epoch_end(self, epoch, logs = {}):
        self.epoch_counter += 1

        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        tr_loss = logs.get('loss')
        tr_acc = logs.get('acc')
        
        with open(self.filepath, "a") as f:
            f.write("{},{},{},{},{}\n".format(self.epoch_counter, tr_loss, tr_acc, val_loss, val_acc))
        
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

def historywriter_factory():
    def factory_f(history_path):
        return HistoryWriter(history_path)
    return factory_f



def callback_factory(callback_type):
    if callback_type == 'model_checkpoint':
        return checkpoint_factory()
    elif callback_type == 'early_stopping':
        return earlystopping_factory()
    elif callback_type == 'lr_kaggle':
        return lrkaggle_factory()
    elif callback_type == 'history_writer':
        return historywriter_factory()
    else:
        raise ValueError("Unknown callback type: " + callback_type)