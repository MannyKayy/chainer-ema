from argparse import ArgumentParser

import chainer
import chainer.links as L
from chainer.training import extensions

from exponential_moving_average import ExponentialMovingAverage
from models import VGG


def run(args):
    model = L.Classifier(VGG())
    train, test = chainer.datasets.get_cifar10()

    if len(args.gpu) == 1 and args.gpu[0] < 0:
        # CPU mode
        pass
    elif len(args.gpu) == 1 and args.gpu[0] >= 0:
        # single GPU mode
        chainer.backends.cuda.get_device_from_id(args.gpu[0]).use()
        model.to_gpu()
    else:
        # multiple GPU mode
        chainer.backends.cuda.get_device_from_id(args.gpu[0]).use()

    optimizer = chainer.optimizers.MomentumSGD(lr=0.05)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    if len(args.gpu) == 1:
        updater = chainer.training.updater.StandardUpdater(train_iter, optimizer, device=args.gpu[0])
    else:
        devices = {str(i): gpu for i, gpu in enumerate(args.gpu[1:])}
        devices['main'] = args.gpu[0]

        updater = chainer.training.updater.ParallelUpdater(train_iter, optimizer, devices=devices)

    trainer = chainer.training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)

    if args.ema_rate != 0.0:
        print('use ema')
        ema = ExponentialMovingAverage(target=model, rate=args.ema_rate, device=args.gpu[0])
        optimizer.add_hook(ema)

        eval_model = ema.shadow_target

        trainer.extend(ema)
    else:
        print('no ema')
        eval_model = model

    # here `eval_model` is passed to the evaluator instead of ordinal `model`
    trainer.extend(extensions.Evaluator(test_iter, eval_model, device=args.gpu[0]))

    # add ordinary extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


def main():
    parser = ArgumentParser(description='Exponential moving decay at chainer')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gpu', '-g', nargs='+', type=int, default=[-1])
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--ema-rate', type=float, default=0.99,
                        help='Exponential moving decay rate. If 0, ema are not applied')
    parser.add_argument('--resume', default='')

    run(parser.parse_args())


if __name__ == '__main__':
    main()
