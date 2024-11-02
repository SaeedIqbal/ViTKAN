import torch
import torch.nn as nn

# Define the Chebyshev polynomial function
def chebyshev_polynomial(x, degree):
    if degree == 0:
        return torch.ones_like(x)
    elif degree == 1:
        return x
    else:
        T0 = torch.ones_like(x)
        T1 = x
        for _ in range(2, degree + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T1

# Define the Legendre polynomial function
def legendre_polynomial(x, degree):
    if degree == 0:
        return torch.ones_like(x)
    elif degree == 1:
        return x
    else:
        P0 = torch.ones_like(x)
        P1 = x
        for n in range(2, degree + 1):
            P2 = ((2 * n - 1) * x * P1 - (n - 1) * P0) / n
            P0, P1 = P1, P2
        return P1

# Define the Kolmogorov-Arnold Network (KAN) class with polynomial functions
class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, poly_degree=3):
        super(KolmogorovArnoldNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_degree = poly_degree

        # Polynomial layers
        self.chebyshev_layer = nn.Linear(input_dim, output_dim)
        self.legendre_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        chebyshev_out = chebyshev_polynomial(self.chebyshev_layer(x), self.poly_degree)
        legendre_out = legendre_polynomial(self.legendre_layer(x), self.poly_degree)
        
        return chebyshev_out + legendre_out
