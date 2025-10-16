import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# === Implémentations perso ===
class SGDPerso:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.v[i] = self.momentum * self.v[i] + g
            p.data -= self.lr * self.v[i]


class AdamPerso:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# === Réseau simple ===
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out)
        )

    def forward(self, x):
        return self.net(x)


def entrainer(model, optim, X, y, epochs=150):
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return losses


# === Expérience ===
torch.manual_seed(42)
N, D_in, D_out = 800, 10, 1
X = torch.randn(N, D_in)
W_true = torch.randn(D_in, D_out)
y = X @ W_true + 0.1 * torch.randn(N, D_out)

hidden = 32
epochs = 200

model_sgd = MLP(D_in, hidden, D_out)
opt_sgd = SGDPerso(model_sgd.parameters(), lr=0.01, momentum=0.9)
loss_sgd = entrainer(model_sgd, opt_sgd, X, y, epochs)

model_adam = MLP(D_in, hidden, D_out)
opt_adam = AdamPerso(model_adam.parameters(), lr=0.01)
loss_adam = entrainer(model_adam, opt_adam, X, y, epochs)

# === Graphiques ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_adam, label="Adam")
plt.title("Évolution de la loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(loss_sgd, label="SGD")
plt.semilogy(loss_adam, label="Adam")
plt.title("Convergence (log)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Loss finale SGD  : {loss_sgd[-1]:.5f}")
print(f"Loss finale Adam : {loss_adam[-1]:.5f}")

# === Analyse perso ===
print("\nAnalyse :")
print("SGD marche bien mais met du temps à stabiliser même avec momentum.")
print("Adam est plus fluide surtout au début et atteint une meilleure loss + vite")
print("Ici les deux finissent proches mais Adam reste plus stable sur la descente.")
