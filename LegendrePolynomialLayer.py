import torch
import torch.nn as nn

class LegendrePolynomialLayer(nn.Module):
    def __init__(self, degree):
        super(LegendrePolynomialLayer, self).__init__()
        self.degree = degree

    def forward(self, x):
        P = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            Pn = ((2 * n - 1) * x * P[-1] - (n - 1) * P[-2]) / n
            P.append(Pn)
        return P[-1]

class LegendreKAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LegendreKAN, self).__init__()
        self.legendre_layer = LegendrePolynomialLayer(degree)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.legendre_layer(x)
        x = self.fc(x)
        return x

# Example usage
model = LegendreKAN(input_dim=10, output_dim=1, degree=5)
input_data = torch.randn(32, 10)
output = model(input_data)
print(output)
