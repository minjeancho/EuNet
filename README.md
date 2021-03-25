# EuNet: Artificial Neural Network with Feedback Connections

This experiment implements Feedback model and Eunet as described here: http://minjeancho.com/eunet/eunet.html. In summary, EuNet is a new NN architecture that is capable of feedback connection, jump connection (like residual net), and interconnection (within a layer).  We compare the  performance of MLP, LSTM, Feedback model, EuNet on the task of memorizing piano notes. 

To train a model, run:
```python
python train.py -m <model_name> -t
```
Where ```python<model_name>``` is the name of the model.

For example:
```python
python train.py -m eunet -t
```

To train all models, run:
```python
python train.py -m all
```

Note that this experiment is in its preliminary stage. This is primarily a proof of concept that the new NN architecture of EuNet is able to memorize sequential data. We chose the task of memorizing piano notes for this reason. In future to test that EuNet learns its own architecture and is a generalization of MLP, Convolution, and RNN, we may conduct the following three experiments: i) Iris dataset: MLP vs EuNet; ii) MNIST dataset: MLP, CNN, vs EuNet; iii) Piano notes: MLP, RNN, vs EuNet. 

