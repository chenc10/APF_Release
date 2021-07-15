# adaptive_freezing
This code is for the APF algorithm described in paper [‘Communication-Efficient Federated Learning with Adaptive Parameter Freezing](https://www.cse.ust.hk/~weiwa/papers/apf-icdcs21.pdf).

## How to run the program
Run `bash train.sh`, which would launch one training process on each worker-ip (can be the same if you want to launch multiple worker processes on one CPU/GPU server). Please read `train.sh` for details.

The logs can be find under `Logs/`, with trial number included in the folder name.

To stop training, run `bash clear.sh <trial-no>` (to stop one particular trial) or `pkill python2` (to stop all the trials).

The first time you run this program, it would take some time to download the CIFAR-10 and MNIST dataset.

Env: torch.__version__ 1.0.0, torchvision.__version__ 0.2.2

## File Description

### worker_process.py
Describes all the steps (load dataset/model, local training iteration, synchronization) of a worker process.

### dataset_manager.py
Specifies the dataset information (iid/noniid, partitions, etc.).

### model_manager.py
Load the specified model defined in `models/` folder.

### sync_manager.py
Handles all the synchronization related issues. Two other sync schemes (Gaia and [CMFL](https://www.cse.ust.hk/~weiwa/papers/cmfl-icdcs19.pdf)) are also implemented for comparison.
