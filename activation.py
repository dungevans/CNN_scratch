import torch 
import numpy as np 
class activation :
    def __init__ (self , act_name : str)  : 
        self.act_name = act_name
        self.input = None
        self.save_input = None  
        self.grad_output = None 
    def forward  (self, input : torch.tensor)-> torch.tensor : 
        self.input = input 
        original_shape = self.input.shape 
        result= np.zeros(shape = original_shape)
        input_flatten = torch.flatten(self.input) 
        input_len =  len(input_flatten)
        self.save_input = input.clone () 
        if self.act_name == 'relu' :
            for i  in range (input_len) : 
                if input_flatten[i] < 0 : 
                    input_flatten[i] = 0 
        self.input = input_flatten.view ( original_shape)
        return self.input
    def backward ( self , grad_output : torch.tensor ) -> torch.tensor :  
        grad_output_original_shape = grad_output.shape
        grad_output_flatten = torch.flatten(grad_output)
        save_input_flatten = torch.flatten(self.save_input)
        grad_input = torch.zeros(len(grad_output_flatten))
        for i in range ( len ( save_input_flatten)) : 
            if save_input_flatten[i] > 0 : 
                grad_input[i] = grad_output[i]
        
        return grad_input.view(grad_output_original_shape)

#test

x = torch.tensor([-2., -1., 0., 1., 2.])
relu = activation('relu')


print("--- BƯỚC 1: FORWARD ---")
out = relu.forward(x)
print("Input gốc x:   ", x)
print("Output (ReLU): ", out)


mock_grad_output = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])

print("\n--- BƯỚC 2: BACKWARD ---")

grad_in = relu.backward(mock_grad_output)

print("Grad Output (Từ lớp sau gửi về):", mock_grad_output)
print("Grad Input  (Lớp ReLU tính ra): ", grad_in)
    
    
