# Random Epipolar Constraint Loss Functions for Supervised Optical Flow Estimation

This repository contains the source code for our paper:
Random Epipolar Constraint Loss Functions for Supervised Optical Flow Estimation.

## Requirements
The code has been tested with PyTorch 1.11.0 and Cuda 10.2.
## Training
After you have checked the code of train.py for the corresponding project, run train.py for training, but before you do that, ensure that the dataset for training is prepared.
Click to enter RECLoss-GMA and run: <br>
                
    ./train.sh
## Evaluation
You can evaluate a trained model using evalu.py, the pretrained weights are available in the 'models' file.

## Acknowledgements
Thanks to [RAFT](https://github.com/princeton-vl/RAFT/tree/master), [GMA](https://github.com/zacjiang/GMA), [GMFlow](https://github.com/haofeixu/gmflow) for providing awesome repos! This has been a great help to us on this project!
