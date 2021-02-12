# FedBS: Learning Batch Statistics for Non-IID Data


## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Datasets
* Datasets will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar-10.

## Reproducing experiments
* To run FedBS with MNIST on CNN with Batch Normalization using CPU (Non-IID):
```
python src/federated_main.py --norm="BN" --model=cnn --dataset=mnist  --iid=0  --epochs=400  --ma="cma" --verbose=0
```
* Or using GPU (if available):
```
python src/federated_main.py --norm="BRN" --model=cnn --dataset=fmnist  --iid=0 --gpu="cuda:0"  --epochs=400  --ma="cma" --verbose=0
```
-----


* To run the baseline experiment with MNIST on CNN with Batch ReNormalization (Non-IID):
```
python src/baseline_main.py --norm="BRN" --model=cnn --dataset=mnist  --iid=0 --gpu="cuda:0"  --epochs=400  --ma="ema-brn" --verbose=0
```


## Options
You can change the following options to simulate different experiments:

* ```--dataset:```  'mnist', 'fmnist' or 'cifar'. Default: 'mnist'.
* ```--model:```    'mlp', 'cnn'. Default: 'cnn'
* ```--gpu:```      "cuda:id" or None(CPU). Default: None.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.
* ```--Norm:```     BN (Batch Normalization) or BRN (Batch Renormalization). Default: BN.
* ```--ma:```     Moving average method used. 'cma', 'ema', 'sma', 'ema-brn'. Default: 'ema'
* ```--resume:```     Default: 0. Set to 1 to resume the training.

* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
