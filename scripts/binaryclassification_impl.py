from cpe487587hw01 import deepl
import matplotlib.pyplot as plt
losses, W1, W2, W3, W4 = deepl.binary_classification(200, 40000, epochs = 50000)

from datetime import datetime

plt.plot(losses.cpu().detach().numpy())

plt.savefig("lossfunction"+str(datetime.now())+".pdf")

print("Training complete.")