import numpy as np

class MaxPoolingLayer:
    def __init__(self, kernel_size):
        """
        @param kernel_size : kernel size for the pooling.
        """
        self.kernel_size = kernel_size
        self.image = None

    def patchGenerator(self, img):
        """
        @param img : Image to be processed.
        @return Patches of image with max applied to it.
        """
        self.image = img
        img_height, img_width = img.shape
        opt_height, opt_width = img_height // self.kernel_size, 
        img_width // self.kernel_size # Get desired output size based on kernel size.
        for h in range(opt_height):
            for w in range(opt_width):
                imgPatch = img[(h * self.kernel_size) : 
                               (h * self.kernel_size + self.kernel_size),
                            (w * self.kernel_size) : 
                            (w * self.kernel_size + self.kernel_size)]
                # Extract patch of image for kernel to be applied on.
                yield imgPatch, h, w

    def forwardProp(self, img):
        """
        @param img: Image to be processed.
        @return Pooled image developed by taking the maximum argument.
        """
        img_height, img_width, n_kernels = img.shape
        opt_shape = (img_height // self.kernel_size, 
                     img_width // self.kernel_size, n_kernels)
        opt = np.zeros(opt_shape) # Initialise output
        for imgPatch, h, w in self.patchGenerator(img):
            opt[h, w] = np.amax(imgPatch, axis = (0, 1)) # Get the argmax for that given patch.
        return opt
    
    def backProp(self, dE_dY):
        """
        @param dE_dY : Gradient taken with respect to the error term.
        @return The loss gradient with respect to the max pooling layer.
        
        Note that:
            dE_dk = dE_dy * dy_dk
            y = argmax(x[i, j]) for i, j in k
            dy_dk = 1 if x[i, j] == max(x) else 0

            Essentially, we only want to replace the max arguments of each patch
            with the corresponding value from dE_dY while ignoring the rest.
        """
        dE_dk = np.zeros(self.image.shape)
        for imgPatch, h, w in self.patchGenerator(self.image):
            patch_height, patch_width, no_kernel = imgPatch.shape
            max_val = np.amax(imgPatch, axis = (0, 1)) # Take the max value in each patch.
            for h_ in range(patch_height):
                for w_ in range(patch_width):
                    for k_ in range(no_kernel):
                        if imgPatch[h_, w_, k_] == max_val: # Check if it is the max value.
                            dE_dk[h * self.kernel_size + h_, 
                                  w * self.kernel_size + w_, k_] = dE_dY[h, w, k_]
        return dE_dk


