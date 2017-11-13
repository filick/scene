import torch.nn as nn
import torch.cuda


class TrainScheme(object):


    def __init__(self, use_multi_gpu=torch.cuda.device_count() > 1):
        self._model = None
        self._train_loader = None
        self._validate_loader = None
        self._criterion = None
        self._optimizer = None
        self._lrschedule = None
        self.use_multi_gpu = use_multi_gpu
        self.hyperparams = {}


    @property
    def model(self):
        if self._model:
            return self._model
        else:
            model = self.init_model()
            if self.use_multi_gpu:
                if self.hyperparams['arch'].lower().startswith('alexnet') or self.hyperparams['arch'].lower().startswith('vgg'):
                    model.features = nn.DataParallel(model.features)
                else:
                    model = nn.DataParallel(model)
            self._model = model
            return self._model


    @property
    def train_loader(self):
        if self._train_loader:
            return self._train_loader
        else:
            self._train_loader, self._validate_loader = self.init_loader()
            return self._train_loader


    @property
    def validate_loader(self):
        self.train_loader
        return self._validate_loader


    @property
    def criterion(self):
        if self._criterion:
            return self._criterion
        else:
            self._criterion = self.init_criterion()
            return self._criterion


    @property
    def optimizer(self):
        if self._optimizer:
            return self._optimizer
        else:
            self._optimizer, self._lrschedule = self.init_optimizer()
            return self._optimizer


    @property
    def lrschedule(self):
        self.optimizer
        return self._lrschedule


    @property
    def name(self):
        raise NotImplementedError("name")

    def init_model(self):
        raise NotImplementedError("model")

    def init_loader(self):
        raise NotImplementedError("loader")

    def init_criterion(self):
        raise NotImplementedError("criterion")

    def init_optimizer(self):
        raise NotImplementedError("optimizer")


class ValidateScheme(object):


    def __init__(self, train_scheme):
        self.model = train_scheme.model
        self.name = train_scheme.name
        self.loader = train_scheme.validate_loader


    def handler(self, model, input_var):
        return model(input_var)