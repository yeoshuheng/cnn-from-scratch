import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_num, kernel_size):
        """
        @param kernel_num : Number of kernels within the layer.
        @param kernel_size : Size of each kernel
        """
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernel = np.random.randn(kernel_num, # Generate kernels (kernel sizes are squared).
                                      kernel_size, kernel_size) / (kernel_size ** 2)
        self.image = None

    def generatePatches(self, img):
        """
        Generate patches of the image which kernels have to be applied on.
        @param img : Image to process
        """
        img_height, img_width = img.shape
        self.image = img

        # Iterates through the image, 
        # kernel_size is being subtracted here due to the 
        # last step in the convolutional process.
        for h in range(img_height - self.kernel_size + 1):
            for w in range(img_width - self.kernel_size + 1):
                imgPatch = img[h : (h + self.kernel_size), w : (w + self.kernel_size)]
                yield imgPatch, h, w # Yield is here to pause execution until results are needed.

    def forwardProp(self, img):
        """
        Conduct forward propogation on the image.
        @param img : Image to process.
        """
        img_height, img_width = img.shape
        # Number of output = number of total sums of matrix multiplication btw patch & kernel.
        output_shape = (img_height - self.kernel_size + 1, 
                        img_width - self.kernel_size + 1,
                        self.kernel_num)
        conv_output = np.zeros(output_shape)
        for imgPatch, h, w in self.generatePatches(img):
            # Iterate through each patch & apply kernel multiplication.
            conv_output[h, w] = np.sum(imgPatch * self.kernel, axis = (1, 2))
        return conv_output

    def backProp(self, dE_dY, alpha):
        dE_dk = np.zeros(self.kernel.shape) # Init dE_dY as zeros
        for imgPatch, h, w in self.generatePatches(self.image):
            patch_h, patch_w = imgPatch.shape
            max_val = np.amax(imgPatch, axis = (0, 1))
            for h_ in range(patch_h):
                for w_ in range(patch_w):
                    for k_ in range(self.kernel_num):
                        if imgPatch[h_, w_, k_] == max_val[k_]:
                            dE_dk[h * self.kernel_size + h_, 
                                  w * self.kernel_size + w_, 
                                  k_] = dE_dY[h, w, k_]
        return dE_dk













