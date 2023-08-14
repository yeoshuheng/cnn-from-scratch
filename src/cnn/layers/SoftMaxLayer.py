import numpy as np

class SoftMaxLayer:
    def __init__(self, input_size, output_size):
        """
        @param input_size : The input size.
        @param output_size : The number of outputs to be assigned.
        """
        # Weights are initialised as random values.
        self.weight = np.random.randn(input_size, output_size) / input_size
        self.bias = np.zeros(output_size) # Initialised bias as zeros.
        self.prev, self.originalsize, self.flattened_image = None, None, None

    def softmax(self, input):
        """
        @param input : Input to be applied to softmax function.
        @return results from the calculation given: 
            e[i] / sum(e[i]) 
        This returns 'probability' for each output by shifting results between [0, 1].
        """
        return np.exp(input) / np.sum(np.exp(input), axis = 0)

    def forwardProp(self, img):
        """
        @param img : Image to be processed.
        @return Probability of given input being of a given output.
        """
        self.originalsize = img.shape
        self.flattened_image = img.flatten()
        # Add weights and biases
        output = np.sum(np.dot(self.flattened_image, self.weight), self.bias)
        self.prev = output # Save previous output
        return self.softmax(output)

    def backProp(self, dE_dY, alpha):
        """
        @param dE_dY : Gradient of the error w.r.t the output.
        @param alpha : Learning rate
        """
        for i, grad in enumerate(dE_dY):
            if grad == 0: # If gradient == 0, nothing to change here.
                continue
            t = np.exp(self.prev)
            total = np.sum(t)
            dY_dZ = -t[i] * t / (total ** 2)
            dY_dZ[i] = t[i] * (total - t[i]) / (total ** 2)

            dZ_dW = self.flattened_image
            dZ_dX = self.weight

            dE_dZ = grad * dY_dZ
            dE_dw = dZ_dW[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * 1 # 1 == dZ_db
            dE_dX = dZ_dX @ dE_dZ

            # Update weights & biases accd. to error gradients.
            self.weight -= alpha * dE_dw
            self.bias -= alpha * dE_db
            
            return dE_dX.reshape(self.originalsize)


            




    