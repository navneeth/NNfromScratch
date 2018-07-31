import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from layers import LinearLayer, LogisticLayer, SoftmaxOutputLayer
from layers import forward_step, backward_step, update_params

if __name__ == "__main__":

    data = datasets.load_iris()

    #create a DataFrame and summarize the dataset
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = pd.DataFrame(data.target)
    print( df.describe() )

    # Defining data and label
    # There are 4 features in this dataset
    X = data.data[:, :4]

    # Load the targets.
    # Note that the targets are stored as labels these need to be
    # converted to one-hot-encoding for the output sofmax layer.
    y = np.zeros((data.target.shape[0],3))
    y[np.arange(len(y)), data.target] += 1
    print('Input shapes are:')
    print('X', X.shape, 'Y', y.shape)

    X, y = shuffle(X, y, random_state=0)

    # Split data into training and test datasets. Since its a small dataset,
    # I select the Traning:Validation:Test = 70:15:15 (i.e training will be based on 70% of data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))

    # Divide the test set into a validation set and final test set.
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Load the standard scaler
    sc = StandardScaler()

    # Compute the mean and standard deviation based on the training data
    sc.fit(X_train)

    # Scale the training data to be of mean 0 and of unit variance
    X_train_std = sc.transform(X_train)

    # Scale the test data to be of mean 0 and of unit variance
    X_test_std = sc.transform(X_test)

    '''
    # Setup a Baseline classifier - SVM - just for comparison
    from sklearn.svm import SVC

    #Applying SVC (Support Vector Classification)

    svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
    svm.fit(X_train, y_train.ravel())
    print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train, y_train)))
    print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test, y_test)))
    '''
    ## 1. Setup Network Architecture
    ########################################################
    # Define a sample model to be trained on the data
    hidden_neurons_1 = 10  # Number of neurons in the first hidden-layer
    hidden_neurons_2 = 10  # Number of neurons in the second hidden-layer
    # Create the model
    layers = [] # Define a list of layers
    # Add first hidden layer
    layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1)) # Inputs -> Hidden Layer
    layers.append(LogisticLayer())

    # Add second hidden layer
    layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
    layers.append(LogisticLayer())

    # Add output layer
    layers.append(LinearLayer(hidden_neurons_2, y_train.shape[1])) # Hidden layer -> Outputs
    layers.append(SoftmaxOutputLayer())

    ## 2. Do a sanity check on the gradients
    ########################################################
    # Perform gradient checking
    nb_samples_gradientcheck = 10 # Test the gradients on a subset of the data
    X_temp = X_train[0:nb_samples_gradientcheck,:]
    T_temp = y_train[0:nb_samples_gradientcheck,:]
    # Get the parameter gradients with backpropagation
    activations = forward_step(X_temp, layers)
    param_grads = backward_step(activations, T_temp, layers)

    # Set the small change to compute the numerical gradient
    eps = 0.0001
    # Compute the numerical gradients of the parameters in all layers.
    for idx in range(len(layers)):
        layer = layers[idx]
        layer_backprop_grads = param_grads[idx]
        # Compute the numerical gradient for each parameter in the layer
        for p_idx, param in enumerate(layer.get_params_iter()):
            grad_backprop = layer_backprop_grads[p_idx]
            # + eps
            param += eps
            plus_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # - eps
            param -= 2 * eps
            min_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # reset param value
            param += eps
            # calculate numerical gradient
            grad_num = (plus_cost - min_cost)/(2*eps)
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_backprop):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)))
    print('No gradient errors found')


    ## 3. Start Training with backpropogation and mini-batch Gradient Descent
    # TODO: 1. Implement stochastic gradient descent instead of plain GD
    # TODO: 2. Add dropout to the layers
    ###########################################################################

    # Perform backpropagation
    # initalize some lists to store the cost for future analysis
    minibatch_costs = []
    training_costs = []
    validation_costs = []

    max_nb_of_iterations = 300 # Train for a maximum of 300 iterations
    learning_rate = 0.1        # Gradient descent learning rate

    # Train for the maximum number of iterations
    for iteration in range(max_nb_of_iterations):

        # Create the minibatches
        batch_size = 25  # Approximately 25 samples per batch
        nb_of_batches = X_train.shape[0] / batch_size  # Number of batches
        # Create batches (X,Y) from the training set
        XT_batches = zip( np.array_split(X_train_std, nb_of_batches, axis=0),  # X samples
                          np.array_split(y_train, nb_of_batches, axis=0))  # Y targets

         # For each minibatch sub-iteration
        for X, T in XT_batches:
            # Calculate a forward pass through the Network & Get the activations
            activations = forward_step(X, layers)

            # Calculate the Error/ cost
            minibatch_cost = layers[-1].get_cost(activations[-1], T)
            minibatch_costs.append(minibatch_cost)

            #backprop and update weights
            param_grads = backward_step(activations, T, layers)  # Get the gradients
            update_params(layers, param_grads, learning_rate)  # Update the parameters

        # Get full training cost for future analysis (plots)
        activations = forward_step(X_train_std, layers)
        train_cost = layers[-1].get_cost(activations[-1], y_train)
        training_costs.append(train_cost)
        # Get full validation cost
        activations = forward_step(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], y_validation)
        validation_costs.append(validation_cost)

        print( 'iteration: {} \
                train loss: {:.3f} \
                validation loss: {:.3f}'.format(iteration, train_cost, validation_cost) )

        if len(validation_costs) > 100:
            # After 100 iterations ,
            # Stop training if the cost on the validation set doesn't decrease
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break


    nb_of_iterations = iteration + 1  # The number of iterations that have been executed


    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    #print( minibatch_x_inds)
    #print( minibatch_costs)
    # Plot the cost over the iterations
    #plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')

    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,nb_of_iterations,0,2.5))
    plt.grid()
    plt.show()

    ## 3. Evaluate Trained network on the Test Set and print statistics
    ###########################################################################

    # Get results of test data
    y_true = np.argmax(y_test, axis=1)  # Get the target outputs
    activations = forward_step(X_test_std, layers)  # Get activation of test samples
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
    print('The accuracy on the test set is {:.2f}'.format(test_accuracy))
    print( metrics.classification_report(y_true, y_pred) )

    # Show confusion table
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0,10)
    minor_ticks = [x + 0.5 for x in range(0, 10)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.suptitle('Confusion table', y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)
    plt.show()
