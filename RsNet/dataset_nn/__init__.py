from easydict import EasyDict as edict

model_mnist_meta = edict({
    'name': 'mnist',
    'width': 28,
    'height': 28,
    'channel': 1,
    'labels': 10
})

model_cifar10_meta = edict({
    'name': 'cifar',
    'width': 32,
    'height': 32,
    'channel': 3,
    'labels': 10
})

model_cifar20_meta = edict({
    'name': 'cifar20',
    'width': 32,
    'height': 32,
    'channel': 3,
    'labels': 20
})

model_cifar100_meta = edict({
    'name': 'cifar100',
    'width': 32,
    'height': 32,
    'channel': 3,
    'labels': 100
})

model_cifar10L_meta = edict({
    'name': 'cifar10L',
    'width': 32,
    'height': 32,
    'channel': 1,
    'labels': 10
})
