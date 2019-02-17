from train import main
import os
import math
import json
import argparse
import numpy as np
import parameters

kernels = [3]
padding = [1]
learning_rates = [0.1, 0.01]
conv_layers = [8, 10]
lin_layers_size = [256]
init_channel = [32]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=False)

    args = parser.parse_args()
    params = parameters.params
    
    start = [0,0,0,0,0]
    try:
        with open('gridsearch.conf', 'r') as f:
            start = [int(k) for k in f.readline().split()]
            f.close()
    except IOError:
        pass
    
    for ker in range(start[0], len(kernels)):
        for lr in range(start[1], len(learning_rates)):
            for cl in range(start[2], len(conv_layers)):
                for lls in range(start[3], len(lin_layers_size)):
                    for ic in range(start[4], len(init_channel)):
                        with open('gridsearch.conf', 'w') as f:
                            f.write(str(ker) + ' ' + str(lr) + ' ' + str(cl) + ' ' + str(lls) + ' ' + str(ic))
                            f.close()
                        args.model_name = str(ker) + str(lr) + str(cl) + str(lls) + str(ic)
                        print("################################")
                        print(args.model_name)
                        print("################################")
                        params["kernel_size"] = conv_layers[cl]*[kernels[ker]]
                        params["padding"] = conv_layers[cl]*[padding[ker]]
                        params["learning_rate"] = learning_rates[lr]
                        params["nb_conv_layers"] = conv_layers[cl]
                        params["lin_layers_size"] = lin_layers_size[lls]
                        params["channel_out"] = [2**(math.floor(1/2*(i+2))-1)*init_channel[ic] for i in range(conv_layers[cl])]
                        params["pool_kernel_size"] = [1 if i%2 == 0 else 2 for i in range(conv_layers[cl])]
                        params["stride"] = [1 for _ in range(conv_layers[cl])]
                        print(params)
                        loss, acc = main(args, params, valid_acc_thresh = 0.72)
                        with open("./models/" + args.model_name + "_acc" + str(acc) + ".params", 'w') as f:
                            json.dump(params, f)
                            f.close()