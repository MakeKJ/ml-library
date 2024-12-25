import numpy as np

class Linear:
    """
    Linear layer in a neural network.
    """

    def __init__(self, x_dim, y_dim):
        """
        Initialize the layer.

        Arguments:
            x_dim: input dimension (int)
            y_dim: output dimension (int)
        """
        # He initialization
        std = np.sqrt(2 / x_dim)
        self.w = np.random.normal(0, std, (y_dim, x_dim))
        self.b = np.zeros(y_dim)

        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass.

        Arguments:
            x: input data (numpy array)

        Returns:
            y: output data (numpy array)
        """ 
        self.x = x
        
        # Handle case where x has more than 2 dimensions
        if len(x.shape) > 2:
            x_reshaped = x.reshape(-1, x.shape[-1])
        else:
            x_reshaped = x
        
        y = np.dot(x_reshaped, self.w.T) + self.b
        
        # Reshape back to the original batch shape if necessary
        if len(x.shape) > 2:
            y = y.reshape(x.shape[0], -1, self.w.shape[0])
        
        return y

    def backward(self, dy):
        """
        Backward pass.

        Arguments:
            dy: gradient of the loss with respect to the output (numpy array)

        Returns:
            dx: gradient of the loss with respect to the input (numpy array)
        """
        self.dw = np.dot(dy.T, self.x)
        self.db = np.sum(dy, axis=0)
        dx = np.dot(dy, self.w)
        return dx


def input_2d_to_columns(x, kernel_size, stride=1, padding=0):
    """
    Converts 4D input tensor to column representation for convolution.
    
    Parameters:
        x:              Input tensor of shape (batch_size, in_channels, height, width).
        kernel_size:    Size of the convolutional kernel.
        stride:         Stride of the convolution.
        padding:        Padding applied to the input.
    
    Returns:
        col:            2D array where each column is an unrolled patch of the input.
    """
    batch_size, in_channels, x_height, x_width = x.shape

    # Add padding to the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Compute output dimensions
    height_out = (x_height + 2 * padding - kernel_size) // stride + 1
    width_out = (x_width + 2 * padding - kernel_size) // stride + 1

    cols = []
    for b in range(batch_size):
        # slices will have dim (in_channels*W_kernel*H_kernel, H_out*W_out)
        slices = []
        for i in range(height_out):
            for j in range(width_out):
                # Extract a slice size of a kernel through all input channels, dim (in_channels, W_kernel, H_kernel)
                slice = x_padded[b, :, 
                                i * stride:i * stride + kernel_size,
                                j * stride:j * stride + kernel_size]

                # Flatten the slice and add it to slices
                slices.append(slice.flatten())
        # Combine slices for this batch example
        cols.append(np.array(slices).T)

    # Combine columns from all batch examples
    col = np.concatenate(cols, axis=-1)  # dim (in_channels*W_kernel*H_kernel, batch_size*H_out*W_out)
    return col


def columns_to_2d_input(col, input_shape, kernel_size, stride=1, padding=0):
    """
    Reconstructs the original 4D input tensor from its column representation.

    Parameters:
        col:            2D array where each column is an unrolled patch of the input.
        input_shape:    Shape of the original input tensor (batch_size, in_channels, height, width).
        kernel_size:    Size of the convolutional kernel.
        stride:         Stride of the convolution.
        padding:        Padding applied to the input.

    Returns:
        x_reconstructed: Reconstructed input tensor of shape `input_shape`.
    """
    batch_size, in_channels, x_height, x_width = input_shape

    # Add padding to the output dimensions
    padded_height = x_height + 2 * padding
    padded_width = x_width + 2 * padding

    # Compute the output dimensions
    height_out = (padded_height - kernel_size) // stride + 1
    width_out = (padded_width - kernel_size) // stride + 1

    # Initialize the padded output tensor
    x_padded = np.zeros((batch_size, in_channels, padded_height, padded_width))

    col_idx = 0  # Column index
    # Iterate over each batch
    for b in range(batch_size):
        for i in range(height_out):
            for j in range(width_out):
                # Extract the flattened patch for this position
                slice = col[:, col_idx].reshape(in_channels, kernel_size, kernel_size)
                col_idx += 1

                # Add the slice to the corresponding location in the padded output tensor
                x_padded[b, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += slice

    # Remove the padding
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    else:
        return x_padded


class Conv2D:
    """
    Convolutional layer in a neural network.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the layer.

        Parameters:
            in_channels:    Number of input channels.
            out_channels:   Number of output channels.
            kernel_size:    Size of the convolutional kernel.
            stride:         Stride of the convolution.
            padding:        Padding applied to the input.
        """
        # Initialize weights using He initialization
        fan_in = in_channels * kernel_size * kernel_size
        self.K = np.random.normal(0, np.sqrt(2 / fan_in), (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels))
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.x_columns = None
        self.kernel_rows = None
        self.db = None
        self.dw = None

    def forward(self, x):
        """Forward pass. Convolution operation performed efficiently using matrix multiplication."""
        batch_size, in_channels, x_height, x_width = x.shape

        # Convert 2D input into column form, dim (C_in*H_kernel*W_kernel, batch_size*W_out*H_out)
        x_columns = input_2d_to_columns(x, self.kernel_size, self.stride, self.padding)

        # Convert kernel to row form, dim (C_out, C_in*H_kernel*W_kernel)
        kernel_rows = self.K.reshape((self.out_channels, -1))

        b_col = self.b.reshape(-1, 1)

        # Matrix multiplication: (C_out, C_in*H_kernel*W_kernel)x(C_in*H_kernel*W_kernel, batch_size*H_out*W_out)
        output_flat = kernel_rows @ x_columns + b_col  # dim (C_out, batch_size*H_out*W_out)

        height_out = (x_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (x_height + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape back to image
        output = np.array(np.hsplit(output_flat, batch_size)).reshape((batch_size, self.out_channels, height_out, width_out))
        
        # Save variables for backpropagation
        self.x = x
        self.x_columns = x_columns
        self.kernel_rows = kernel_rows
        return output
        
    def backward(self, dy):
        """Backward pass. Convolution operation performed efficiently using matrix multiplication."""
        batch_size, out_channels, _, _ = dy.shape
        out_channels, in_channels, kernel_height, kernel_width = self.K.shape
        x_columns, kernel_rows = self.x_columns, self.kernel_rows

        # Partial derivative wrt biases the sum over spatial dimensions
        self.db = np.sum(dy, axis=(0, 2, 3))

        # Reshape dy
        dy = dy.reshape(batch_size * out_channels, -1)  # dim (batch_size*C_out, W_out*H_out)
        dy = np.array(np.vsplit(dy, batch_size))  # dim (batch_size, C_out, W_out*H_out)
        dy = np.concatenate(dy, axis=-1)  # dim (C_out, batch_size*W_out*H_out)

        # Partial derivatives wrt kernel weights
        # Matrix multiplication: (C_out, batch_size*W_out*H_out)x(batch_size*W_out*H_out, C_in*H_kernel*W_kernel)
        dw_col = dy @ x_columns.T  # dim (C_out, C_in*H_kernel*W_kernel)
        self.dw = dw_col.reshape(out_channels, self.in_channels, kernel_height, kernel_width)

        # Partial derivatives wrt inputs
        # Matrix multiplication: (C_in*H_kernel*W_kernel, C_out)x(C_out, batch_size*W_out*H_out)
        dx_col = kernel_rows.T @ dy  # dim (C_in*H_kernel*W_kernel, batch_size*H_out*W_out)

        # Convert dx_col to image form
        dx = columns_to_2d_input(dx_col, self.x.shape, self.kernel_size, self.stride, self.padding)
        return dx
        

class MaxPool2D:
    """
    Max pooling layer in a neural network.
    """
    def __init__(self, pool_size, stride):
        """
        Initialize the layer.

        Parameters:
            pool_size:  Size of the pooling window.
            stride:     Stride of the pooling operation.
        """
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.x_columns = None
        self.max_indices = None

    def forward(self, x):
        """Forward pass."""
        self.x = x
        batch_size, in_channels, x_height, x_width = x.shape
        pool_size, stride = self.pool_size, self.stride

        height_out = (x_height - pool_size) // stride + 1
        width_out = (x_width - pool_size) // stride + 1

        # Convert 2D input into column form
        x_columns = input_2d_to_columns(x, pool_size, stride)  # dim (C_in*pool_size*pool_size, batch_size*H_out*W_out)
        self.x_columns = x_columns
        x_columns = x_columns.reshape(in_channels, x_columns.shape[0] // in_channels, -1)  # dim (C_in, pool_size*pool_size, batch_size*H_out*W_out)
        
        # Save max indices for backpropagation
        self.max_indices = np.argmax(x_columns, axis=1)  # dim (C_in, batch_size*H_out*W_out)

        # Compute and reshape the output
        output = np.max(x_columns, axis=1)
        output = np.array(np.hsplit(output, batch_size)).reshape((batch_size, in_channels, height_out, width_out))
        return output

    def backward(self, dy):
        """Backward pass."""
        batch_size, in_channels, height_out, width_out = dy.shape
        pool_size, stride = self.pool_size, self.stride
        
        # Initialize dx_columns to zeros
        dx_columns = np.zeros_like(self.x_columns)  # dim (C_in*pool_size*pool_size, batch_size*H_out*W_out)

        # Reshape dy
        dout_reshaped = dy.transpose(1, 2, 3, 0).reshape(in_channels, height_out * width_out, batch_size)

        # Reshape max_indices
        max_indices_reshaped = self.max_indices.reshape(in_channels, batch_size, height_out * width_out)

        # Inset the gradients to the correct positions using max_indices
        for i in range(batch_size):
            for j in range(height_out * width_out):
                for c in range(in_channels):
                    # Get the index of the max value from max_indices
                    max_idx = max_indices_reshaped[c, i, j]
                    grad_output = dout_reshaped[c, j, i]

                    # Compute indices in dx_columns
                    channel_idx = max_idx + c * pool_size * pool_size
                    batch_index = i * height_out * width_out + j

                    # Add the gradient to the appropriate location in dx_columns
                    dx_columns[channel_idx, batch_index] += grad_output

        # Reconstruct the gradient wrt input using
        dx = columns_to_2d_input(dx_columns, self.x.shape, pool_size, stride)
        return dx