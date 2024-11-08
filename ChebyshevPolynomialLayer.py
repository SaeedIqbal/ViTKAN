import torch
import torch.nn as nn

class ChebyshevPolynomialLayer(nn.Module):
    def __init__(self, degree):
        super(ChebyshevPolynomialLayer, self).__init__()
        self.degree = degree

    def forward(self, x):
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            Tn = 2 * x * T[-1] - T[-2]
            T.append(Tn)
        return T[-1]

class ChebyKAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKAN, self).__init__()
        self.cheby_layer = ChebyshevPolynomialLayer(degree)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.cheby_layer(x)
        x = self.fc(x)
        return x

# Example usage
model = ChebyKAN(input_dim=10, output_dim=1, degree=5)
input_data = torch.randn(32, 10)
output = model(input_data)
print(output)
