# NNfromScratch

Coding a Neural Network with no libraries. (except numpy ;) ). An exploration from first principles.
At the moment, a simple network with 1 hiddenlayer is implemented.

It implements:
* A layer to apply the linear transformation (LinearLayer).
* A layer to apply the logistic function (LogisticLayer).
* A layer to compute the softmax classification probabilities at the output (SoftmaxOutputLayer).
Each layer can compute its output in the forward step with get_output, which can then be used as the input for the next layer. The gradient at the input of each layer in the backpropagation step is computed with get_input_grad.

The last layer contains a softmax over the output.
* Optimization of the loss is using a mini-batch gradient descent

Usage:
```python
pip install -r requirements.txt
python main.py
```

Sample output:
The code displays the loss trend on the training and validation set.
It also displays the accuray and confusion matrix on the test set.

Currently:
The accuracy on the test set is 0.70
