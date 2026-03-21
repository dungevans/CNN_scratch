import numpy as np
import time
    
# class CNN3D : 
#     def __init__ (self, input_channel : int ,   output_channel : int , kernel : int , Stride : int , Padding : int    )  :  
#          self.input_channel = input_channel
#          self.output_channel = output_channel 
#          self.Stride = Stride 
#          self.Padding = Padding 
#          self.kernel = kernel 
#          self.kernel_matrix = np.random.rand(output_channel, input_channel, kernel , kernel )  
#          self.bias =np.random.randn (output_channel)
     
         
         
#     def forward (self , x :np.array ):
#          #x.shape = (a,b,3)
#         if self.Padding > 0:
#             p = self.Padding
#             x= np.pad(x, ((p, p), (p, p), (0, 0)), mode='constant')
#         H_out = ((x.shape[0]-self.kernel)/self.Stride) + 1 
#         W_out = ((x.shape[1]-self.kernel)/self.Stride) + 1 
       

#         result = np.zeros(( self.output_channel, int(H_out),int  (W_out)) ) 
#         for output in range ( self.output_channel) : 
            
#                 output_j = 0 
#                 for j in range (0,x.shape[1]-self.kernel+1,self.Stride ) :
#                     output_i = 0 
#                     for i in range ( 0, x.shape[0]-self.kernel+1,self.Stride) :
                     
#                         kernel_slice_matrix =self.kernel_matrix[output].transpose(1, 2, 0)
#                         sub_matrix_x = x[i:i+self.kernel,j:j+self.kernel,:]
#                         result1 = sub_matrix_x * kernel_slice_matrix
                        
#                         single_value = np.sum(result1) +self.bias[output] 
#                         result[output_i,output_j,output] += single_value 
#                         output_i +=1   
#                     output_j +=1    
#         return result             

class Conv2D: 
    def __init__(self, in_channel: int, out_channel: int, kernel: int, stride: int, padding: int):
        self.in_channel = in_channel 
        self.out_channel = out_channel
        self.kernel = kernel 
        self.stride = stride 
        self.padding = padding 
        self.W = np.random.randn(out_channel, in_channel, kernel, kernel)
        self.bias = np.random.randn(out_channel)
        self.x_pad = None
        self.weight_grad = None
        self.bias_grad = None
          
    def forward(self, x): 
        """
        x.shape   (batch, inchannel, H ,W )
        """
        p = self.padding 
        if p >0 : 
            self.x_pad= np.pad(x, ((0, 0), (0,0), (p, p), (p, p)), mode='constant')
        N, C_in, H_in, W_in = self.x_pad.shape     
        H_out = int(((H_in - self.kernel) / self.stride) + 1)
        W_out = int(((W_in - self.kernel) / self.stride) + 1)
        result = np.zeros((N,self.out_channel, H_out, W_out))
        for n in range (N) : 
            for c in range(self.out_channel): 
                out_j = 0 
                for j in range(0, W_in - self.kernel + 1, self.stride): 
                    out_i = 0 
                    for i in range(0, H_in - self.kernel + 1, self.stride):
                        
                    
                        x_slices = self.x_pad[n,:, i : i+self.kernel, j : j+self.kernel]
                        x_slices_flat = x_slices.flatten() 
                        
                        
                        kernel_c = self.W[c, :, :, :]
                        
                        kernel_c_flat = kernel_c.flatten()
                        

                        a = np.dot(x_slices_flat, kernel_c_flat) + self.bias[c]
                    
                        result[n,c, out_i, out_j] = a  
                        
                        out_i += 1 
                    out_j += 1 
                
        return result
    def backward(self, dy : np.array ) :
        "dy.shape(batch_size, out_channel, H_out, W_out )"
        dx_pad = np.zeros_like(self.x_pad)
        
        
        self.weight_grad = np.zeros_like(self.W)
        self.bias_grad = np.zeros_like(self.bias)

        batch_dy, ochannel_dy, H_dy, W_dy = dy.shape  
        batch_dx, ochannel_dx, H_dx, W_dx = self.x_pad.shape
        for n in range ( batch_dy ) :
            for c in range ( ochannel_dy) : 
                for j in range (0,H_dy-self.kernel+1, self.stride) :
                    for i in range (0, W_dy-self.kernel+1 , self.stride ) :
                        x_slices = self.x_pad[n,:, i : i+self.kernel, j : j+self.kernel]
                        x_slices_flat = x_slices.flatten() 
                        self.weight_grad[c, : , : , : ]+= x_slices*dy[n,c,j, i ]
                        dx_pad[n,:, i : i+self.kernel, j : j+self.kernel] += self.W[n,:, :, : ]*dy[n,c,j, i ]






                self.bias_grad[c] =  np.sum ( dy[n,c,:, : ])     
                     

    def step ( self,lr  ): 
        self.W = self.W - lr*self.weight_grad
        self.bias = self.bias -lr*self.bias_grad
    def zero_grad ( self ) : 
        self.weight_grad = None 
        self.bias_grad = None 

class Linear : 
    def __init__ ( self, input_size, output_size ) : 

        self.input_size = input_size 
        
        self.input = None # bo nho dem 
        self.output_size = output_size 
        self.W = np.random.randn(self.output_size, self.input_size )
        self.bias =np.random.randn (self.output_size) 
        self.w_gr = None 
        self.b_gr =None
    def forward ( self, data ) : 
        #data la 1 tensor 1 chieu 
        self.input = data 
        result = np.dot(data,self.W.T) + self.bias
        return result 
    def backward(self, dy : np.array ) :
        #dy(bz, output)
        #dx(bz,input)
        self.w_gr = np.dot(dy.T,self.input)
        self.b_gr = np.sum(dy, axis = 0 )
        input_gr = np.dot(dy,self.W)
        return input_gr
    def step(self, learning_rate: float):
     
        
        self.W -= learning_rate * self.w_gr
        self.bias -= learning_rate * self.b_gr
        
    def zero_grad(self):
        
        self.w_gr = None
        self.b_gr = None
        
        



class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: Đầu ra thô từ lớp cuối cùng của mạng (batch_size, num_classes)
        y_true: Nhãn thực tế dạng one-hot encoding (batch_size, num_classes)
        """
      
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true

        batch_size = logits.shape[0]
        
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1 - 1e-7)
        
        
        loss = -np.sum(self.y_true * np.log(y_pred_clipped)) / batch_size
        return loss

    def backward(self):
        """
        Tính đạo hàm của Loss truyền ngược về lớp trước đó.
        """
        batch_size = self.y_true.shape[0]
        
        
        dy = (self.y_pred - self.y_true) / batch_size
        return dy



class SGD:
    def __init__(self, layers, lr=0.01):
        """
        layers: Danh sách (list) các lớp trong mạng nơ-ron (Conv2D, Linear,...)
        lr: Learning rate (Tốc độ học)
        """
        self.layers = layers
        self.lr = lr

    def step(self):
        """
        Cập nhật trọng số cho tất cả các lớp có chứa tham số học (W và bias).
        """
        for layer in self.layers:
            
            if hasattr(layer, 'W') and hasattr(layer, 'weight_grad'):
                layer.W = layer.W - self.lr * layer.weight_grad
                layer.bias = layer.bias - self.lr * layer.bias_grad

    def zero_grad(self):
        """
        Xóa gradient (đặt về 0) trước mỗi vòng lặp forward/backward mới.
        """
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()





import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dy):
        return dy * (self.input > 0)

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        """
        x: (Batch_size, Channels, Height, Width)
         return (Batch_size, Channels * Height * Width)
        """
        self.input_shape = x.shape
        batch_size = x.shape[0]
        
        return x.reshape(batch_size, -1)

    def backward(self, dy):
        """
        dy: (Batch_size, Flattened_Dim)
         (Batch_size, Channels, Height, Width)
        """
        
        return dy.reshape(self.input_shape)
    
class CNN:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def backward(self, dy):
        grad = dy
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad














import time


np.random.seed(42) 
batch_size = 4    
img_ch, img_h, img_w = 3, 32, 32 
num_classes = 2
learning_rate = 0.001
epochs = 5


X_train = np.random.randn(20, img_ch, img_h, img_w) 
y_train_raw = np.random.randint(0, num_classes, 20)

y_train = np.eye(num_classes)[y_train_raw]



def get_flatten_dim(input_shape, conv_layers):
    dummy_x = np.zeros((1, *input_shape))
    out = dummy_x
    for layer in conv_layers:
        out = layer.forward(out)
    return out.size 


conv_branch = [
    Conv2D(in_channel=3, out_channel=8, kernel=3, stride=2, padding=1), 
    ReLU()
]


flatten_dim = get_flatten_dim((img_ch, img_h, img_w), conv_branch)
print(f"Features after Flatten: {flatten_dim}")


model_layers = conv_branch + [
    Flatten(),
    Linear(flatten_dim, 64),
    ReLU(),
    Linear(64, num_classes)
]

cnn_model = CNN(model_layers)
criterion = CrossEntropyLoss()
optimizer = SGD(model_layers, lr=learning_rate)


num_samples = X_train.shape[0]

for epoch in range(epochs):
    start_time = time.time()
    epoch_loss = 0
    
    
    for i in range(0, num_samples, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        
        optimizer.zero_grad()
        
        
        logits = cnn_model.forward(X_batch)
        
        
        loss = criterion.forward(logits, y_batch)
        epoch_loss += loss * X_batch.shape[0]
        
        
        dy = criterion.backward()
        cnn_model.backward(dy)
        
        
        optimizer.step()
        
    avg_loss = epoch_loss / num_samples
    end_time = time.time()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Time: {end_time-start_time:.2f}s")




