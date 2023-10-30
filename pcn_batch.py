import numpy as np

class pcn:
    
    def __init__(self,inputs,targets):
        """ Constructor """
        # Set up network size
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
    
        # Initialise network
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self,inputs,targets,eta,nIterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
    
        # Training
        for n in range(nIterations):
            for m in range(self.nOut):
                index = n % np.shape(inputs)[0]
                activation = sum(self.weights[:, m] * inputs[index])

                activation = 1 if activation > 0 else 0

                self.weights[:, m] += eta * (targets[index][m] - activation) * inputs[index]

                print("\nIteration:", n)
                print("Target Output:", targets[index][m])
                print("Predicted Output:", activation)
                print("Weights:")
                print(self.weights)

    def confmat(self,inputs,targets):
        """Confusion matrix"""
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)
    
        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum((outputs == i) & (targets == j))

        print(cm)
        print(np.trace(cm) / np.sum(cm))