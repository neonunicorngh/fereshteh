#!/bin/bash
echo "Experiment 1: Small dataset, few epochs"
python binaryclassification_impl_cmdline.py -d 100 -n 10000 -e 500

echo "Experiment 2: Medium dataset, medium epochs"
python binaryclassification_impl_cmdline.py -d 200 -n 25000 -e 1000

echo "Experiment 3: Large dataset, many epochs"
python binaryclassification_impl_cmdline.py -d 300 -n 50000 -e 2000

echo "All experiments completed!"
