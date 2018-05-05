import numpy as np
from chainer import cuda, Link, Chain, ChainList
from chainer.training import extension


def _namedpersistents_as_link(target):
    assert isinstance(target, Link)

    d = target.__dict__
    for name in target._persistent:
        yield '/' + name, d[name]


def _namedpersistents_as_chain(target):
    assert isinstance(target, Chain)

    for name, persistent in _namedpersistents_as_link(target):
        yield name, persistent

    d = target.__dict__
    for name in target._children:
        prefix = '/' + name
        for path, persistent in namedpersistents(d[name]):
            yield prefix + path, persistent


def _namedpersistents_as_chain_list(target):
    assert isinstance(target, ChainList)

    for name, persistent in _namedpersistents_as_link(target):
        yield name, persistent

    for idx, link in enumerate(target._children):
        prefix = '/%d' % idx
        for path, persistent in namedpersistents(link):
            yield prefix + path, persistent


def namedpersistents(target):
    if isinstance(target, Chain):
        retriever = _namedpersistents_as_chain
    elif isinstance(target, ChainList):
        retriever = _namedpersistents_as_chain_list
    elif isinstance(target, Link):  # do not put this above, because Chain/ChainList are Link
        retriever = _namedpersistents_as_link
    else:
        raise ValueError

    for name, persistent in retriever(target):
        yield name, persistent


class ExponentialMovingAverage(extension.Extension):

    name = 'ExponentialMovingAverage'
    timing = 'post'

    def __init__(self, target, rate, device=None):
        self.shadow_target = target.copy()
        self._shadow_data = dict()

        self._rate = rate
        self._device = device

        self._initialized = False
        self._param_names = set()

        for name, _ in target.namedparams():
            self._param_names.add(name)

    def __call__(self, optimizer):
        if not self._initialized:
            self._initialize()

        target_persistents = {}
        for name, param in namedpersistents(optimizer.target):
            target_persistents[name] = param

        # copy all persistents to shadow_target
        # without this, all of persistents in shadow_target will be initialized in multiprocessing environments
        for name, persistent in namedpersistents(self.shadow_target):

            # persistent's type is numpy/cupy array or scalar (int/float)
            if isinstance(persistent, cuda.ndarray):
                persistent.data.copy_from(target_persistents[name].data, persistent.size * persistent.dtype.itemsize)
            else:
                persistent = target_persistents[name]

        for name, param in optimizer.target.namedparams():
            self._update_shadow(name, param)

        for name, param in self.shadow_target.namedparams():
            param.data = self._shadow_data[name]

    @property
    def trigger(self):
        return None

    def _initialize(self):
        # necessary for cases when using multiprocess parallel updater
        self.shadow_target.to_gpu(self._device)
        self._initialized = True

    def _update_shadow(self, name, param):
        s, p = self._shadow_data.get(name), param.data

        if p is None:
            return

        if s is None:
            self._shadow_data[name] = cuda.get_array_module(p).array(p)
            return

        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                s -= (1 - self._rate) * (s - p)

            else:
                kernel = cuda.elementwise('T p, T decay',
                                          'T s',
                                          's -= (1 - decay) * (s - p)',
                                          'exponential_moving_average')
                kernel(p, self._rate, s)

    def serialize(self, serializer):
        for name in self._param_names:
            shadow_data = self._shadow_data.get(name)
            data = serializer['shadow_params'](name, shadow_data)

            if shadow_data is None and data is not None:
                if self._device == -1:
                    self._shadow_data[name] = np.array(data)
                else:
                    self._shadow_data[name] = cuda.to_gpu(data, device=self._device)
