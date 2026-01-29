#!/usr/bin/env python

import sys
import getopt

from cpe487587hw01 import deepl
import matplotlib.pyplot as plt

from datetime import datetime

def main(argv):
    
    d_features = 200
    n_samples = 40000
    e_epochs = 5000
      
    try:
        opts, args = getopt.getopt(argv, "hd:n:e:", ["d_features=", "n_samples=", "epochs="])
        #if len(opts) == 0:
        #    print('Check options by typing:\n{} -h'.format(__file__))
        #    sys.exit()

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -d, --d_features\t Number of features')
            print('\t -n, --n_samples\t Number of samples')
            print('\t -e, --epochs\t\t Number of training epochs')
            sys.exit()
        elif opt in ("-d", "--d_features"):
            d_features = eval(arg)
        elif opt in ("-n", "--n_samples"):
            n_samples = eval(arg)
        elif opt in ("-e", "--epochs"):
            epochs = eval(arg)
    
    losses, W1, W2, W3, W4 = deepl.binary_classification(d_features, n_samples, epochs = e_epochs)

    
    plt.plot(losses.cpu().detach().numpy())
    plt.savefig("lossfunction_" + str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".","__") + ".pdf")

    print("Training complete.")



if __name__ == "__main__":
   main(sys.argv[1:])
