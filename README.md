# MarginGAN
This repository is the implementation of the paper "MarginGAN: Adversarial Training in Semi-Supervised Learning".

1."preliminary" is the implementation of **Preliminary Experiment on MNIST** of the paper. Thank the authors of [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections) and [examples of pytorch](https://github.com/pytorch/examples/blob/master/mnist/main.py), our code is widely adapted from their repositories.

To train the network, an example is as follows:
```
python3 main.py \
  --gan_type MarginGAN \
  --num_labels 600 
  --lrC 0.1 
  --epoch 50
```  
2."Combined with Mean Teacher" is the implementation of **Experiment on SVHN and CIFAR-10** of the paper. Thank the authors of [mean teacher](https://github.com/CuriousAI/mean-teacher), our code is widely adapted from their repositories.

To train the network, an example is as follows:
```
python MarginGAN_main.py \
    --dataset cifar10 \
    --train-subdir train+val \
    --eval-subdir test \
    --batch-size 128 \
    --labeled-batch-size 31 \
    --arch cifar_shakeshake26 \
    --consistency-type mse \
    --consistency-rampup 5 \
    --consistency 100.0 \
    --logit-distance-cost 0.01 \
    --weight-decay 2e-4 \
    --lr-rampup 0 \
    --lr 0.05 \
    --nesterov True \
    --labels data-local/labels/cifar10/1000_balanced_labels/00.txt  \
    --epochs 180 \
    --lr-rampdown-epochs 210 \
    --ema-decay 0.97 \
    --generated-batch-size 32
```
