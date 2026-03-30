import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(Normalize, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps
        
    def forward(self, data_origin):
        batch_size = data_origin.shape[0]
        self.data_origin = data_origin 
        self.data_normalized = torch.zeros_like(data_origin)
        self.means = torch.zeros(batch_size, 1, 1, 1, device=data_origin.device)
        self.vars = torch.zeros(batch_size, 1, 1, 1, device=data_origin.device)

        for i in range(batch_size):
            data = data_origin[i]
            mean_i = data.mean()
            var_i = data.var(unbiased=False)
            
            data_z_score = (data - mean_i) / torch.sqrt(var_i + self.eps)
            self.data_normalized[i] = self.gamma * data_z_score + self.beta
            
            self.means[i] = mean_i
            self.vars[i] = var_i

        return self.data_normalized
    def backward(self, d_output: torch.Tensor):
      
        batch_size, C, H, W = d_output.shape
        M = C * H * W 
        
       
        dx = torch.zeros_like(d_output)
        
      
        self.grad_gamma = torch.sum(d_output * self.data_normalized)
       
        self.grad_beta = torch.sum(d_output)

    
        for i in range(batch_size):
            
            d_y_i = d_output[i] * self.gamma
            x_hat_i = self.data_normalized[i]
            std_inv_i = 1.0 / torch.sqrt(self.vars[i] + self.eps)

            term1 = M * d_y_i
            term2 = torch.sum(d_y_i)
            term3 = x_hat_i * torch.sum(d_y_i * x_hat_i)
            
            dx[i] = (1.0 / M) * std_inv_i * (term1 - term2 - term3)

        return dx
def test_normalize_layer():
    # 1. Khởi tạo dữ liệu giả lập (Batch=2, C=3, H=4, W=4)
    input_data = torch.randn(2, 3, 4, 4, requires_grad=True)
    
    # 2. Khởi tạo lớp của bạn
    my_norm = Normalize(channels=3)
    
    # 3. Khởi tạo LayerNorm của PyTorch để đối chiếu
    # Lưu ý: LayerNorm tính trên toàn bộ các chiều ngoại trừ Batch (C, H, W)
    torch_norm = nn.LayerNorm(input_data.shape[1:]) 
    
    # Copy trọng số để đảm bảo so sánh công bằng
    with torch.no_grad():
        torch_norm.weight.fill_(1.0)
        torch_norm.bias.fill_(0.0)

    # --- TEST FORWARD ---
    my_output = my_norm(input_data)
    torch_output = torch_norm(input_data)
    
    # Kiểm tra sai số giữa 2 kết quả (Sai số cho phép cực nhỏ 1e-6)
    forward_diff = torch.abs(my_output - torch_output).max().item()
    print(f"Forward Difference: {forward_diff:.8f}")
    
    if forward_diff < 1e-6:
        print("✅ Forward Pass: TRÙNG KHỚP")
    else:
        print("❌ Forward Pass: SAI LỆCH")

    # --- TEST BACKWARD ---
    # Giả sử ta có một giá trị Loss đơn giản là tổng các phần tử output
    my_output.sum().backward()
    
    # Lấy gradient của input từ lớp của bạn (nếu bạn dùng autograd của torch)
    my_grad = input_data.grad.clone()
    
    # Reset grad và tính toán với torch_norm
    input_data.grad.zero_()
    torch_output.sum().backward()
    torch_grad = input_data.grad
    
    backward_diff = torch.abs(my_grad - torch_grad).max().item()
    print(f"Backward Difference: {backward_diff:.8f}")

    if backward_diff < 1e-6:
        print("✅ Backward Pass: TRÙNG KHỚP")
    else:
        print("❌ Backward Pass: SAI LỆCH")

if __name__ == "__main__":
    test_normalize_layer()