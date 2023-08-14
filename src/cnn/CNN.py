import numpy as np
import layers

class CNN:
    def __init__(self, layers_, alpha = 0.05):
        self.layers = layers_
        self.alpha = alpha

    def forwardProp(self, img, label):
        """
        @param img : Input image to be processed.
        @param label : Target variable.
        @return predicted prob (output), gap between targeted and predicted (loss) 
            and accuracy (acc)
        """
        output  = img / 255
        for layer in self.layers:
            output = layer.forwardProp(output)
        loss = -np.log(output[label]) # Calculate loss.
        acc = 1 if (np.argmax(output) == label) else 0 # Check if target met.
        return output, loss, acc

    def backprop(self, grad):
        """
        @param grad : Gradient from pass layers.
        @return backpropogated gradient.
        """
        grad_bp = grad
        for layer in self.layers[::-1]: # Ignore output layer.
            grad_bp = layer.backProp(grad_bp, self.alpha)
        return grad_bp
    
    def trainModel(self, img, label, n_classes):
        """
        Does one pass of forward and back propogation. 
        @param img : Image to process.
        @param label : Label of the image.
        @param n_classes : Number of target classes
        """
        output, loss, acc = self.forwardProp(img, label)
        grad = np.zeros(n_classes)
        grad[label] = - 1 / output[label] # Obtain gradient of predicted label.
        grad_bp = self.backprop(grad) # Backpropogate to train model.
        return loss, acc





