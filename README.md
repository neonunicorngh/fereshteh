# Homework 01 CPE 487 587


If you want to create this repo from scratch without cloning, go to **Step 1**. Otherwise, go to **Step A**.

## Step A

```
git clone https://github.com/rahulbhadani/cpe487587hw01
cd cpe487587hw01
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```

-----------------------------------------------------------------------------------------------------------------------------


## Step 1

```
uv init --lib --package cpe487587hw01 --build-backend maturin --author-from git
cd cpe487587hw01
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

## Step 2

```
cd src/cpe487587hw01
mkdir deepl
cd deepl
touch two_layer_binary_classification.py
vi two_layer_binary_classification.py
```

and add the following code

```python
import torch

def binary_cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-7  # to Prevent log(0) or log(1)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss

def binary_classification(d, n, epochs = 10000, eta = 0.001):
    """
    Binary Classification with Linear and Nonlinear Layers
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(axis=1, keepdim=True) > 2).float()

    # W1 = torch.normal(mean = 0, std = torch.sqrt(torch.tensor(d)), size = (d, 48), requires_grad=True, dtype=torch.float32, device=device)

    current_dtype = torch.float32
    W1 = (torch.randn(d, 48, device=device, dtype=current_dtype) * torch.sqrt(torch.tensor(1.0/d, device=device, dtype=current_dtype))).requires_grad_(True)
    W2 = (torch.randn(48, 16, device=device, dtype=current_dtype) * torch.sqrt(torch.tensor(1.0/48, device=device, dtype=current_dtype))).requires_grad_(True)
    W3 = (torch.randn(16, 32, device=device, dtype=current_dtype) * torch.sqrt(torch.tensor(1.0/16, device=device, dtype=current_dtype))).requires_grad_(True)
    W4 = (torch.randn(32, 1, device=device, dtype=current_dtype) * torch.sqrt(torch.tensor(1.0/32, device=device, dtype=current_dtype))).requires_grad_(True)

    train_losses = torch.zeros(epochs, device=device)

    for epoch in range(epochs):
        Z1 = torch.matmul(X, W1)
        Z1 = torch.matmul(Z1, W2)
        A1 = 1/(1 + torch.exp(-Z1))
        Z2 = torch.matmul(A1, W3)
        Z2 = torch.matmul(Z2, W4)
        A2 = 1/(1 + torch.exp(-Z2))
        YPred = A2

        train_loss = binary_cross_entropy_loss(YPred, Y)

        train_loss.backward()

        with torch.no_grad():
            W1 -= eta*W1.grad
            W2 -= eta*W2.grad
            W3 -= eta*W3.grad
            W4 -= eta*W4.grad

            # Zero the gradients
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()

            train_losses[epoch] = train_loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {train_loss.item():.4f}")

    return [train_losses, W1, W2, W3, W4]

```

## Step 3


```
cd ..
vi __init__.py
```

Add following to `__init__.py` of the subpackage

```bash
from .two_layer_binary_classification import binary_classification
```

## Step 4

```
cd ..
vi __init__.py
```
Add following to `__init__.py` of the package

```
from .deepl import binary_classification
```

## Step 5
Go to the root directory of the UV project

```
cd ../..
```

Add dependencies

```
uv add torch numpy matplotlib
```

Build the package

```
uv build
```

## Step 6

Create a scripts folder

```
cd scripts
vi binaryclassification_impl.py
```


Add the following code

```python
from cpe487587hw01 import deepl
losses, W1, W2, W3, W4 = deepl.binary_classification(200, 40000, epochs = 50000)

print("Training complete.")
```

## Automating using bash script

Create a bashcript binary_class.sh and the following code:

```bash
#!/bin/bash

# Define the script name
SCRIPT_NAME="binaryclassification_impl.py"

echo "Starting the binary classification training task..."

# Check if the file exists before running
if [ -f "$SCRIPT_NAME" ]; then
    python3 "$SCRIPT_NAME"
else
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi

echo "Process finished at $(date)"
```

Make it executable

```
chmod +x binary_class.sh
```

To run in the background:


```
nohup ./binary_class.sh > training_log.out 2>&1 &
```


`nohup` Prevents the process from being killed when you close the terminal.

`./run_training.sh` Executes your automation script.

`> training_log.out` Redirects standard output (logs) to a file so you can check progress.

`2>&1` Redirects error messages to the same log file.

`&` Puts the entire process in the background.


**END**


## HW02Q7

### Setup
From the project root:

```bash
uv sync
uv add manim
uv run python scripts/binaryclassification_animate_impl.py
media/videos/1080p30/
 