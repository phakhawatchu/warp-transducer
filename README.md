# warp-transducer (worked on WSL2)

A fast parallel implementation of RNN Transducer (Graves 2013 joint network), on both CPU and GPU.

[GPU implementation is now available for Graves2012 add network.](https://github.com/HawkAaron/warp-transducer/tree/add_network)

## GPU Performance

Benchmarked on a GeForce GTX 1080 Ti GPU.

| **T=150, L=40, A=28** | **warp-transducer** |
| --------------------- | ------------------- |
| N=1                   | 8.51 ms             |
| N=16                  | 11.43 ms            |
| N=32                  | 12.65 ms            |
| N=64                  | 14.75 ms            |
| N=128                 | 19.48 ms            |

| **T=150, L=20, A=5000** | **warp-transducer** |
| ----------------------- | ------------------- |
| N=1                     | 4.79 ms             |
| N=16                    | 24.44 ms            |
| N=32                    | 41.38 ms            |
| N=64                    | 80.44 ms            |
| N=128                   | 51.46 ms            |

<!-- | **T=1500, L=300, A=50** | **warp-transducer** |
| ----------------------- | ------------------- |
|         N=1             |      570.33 ms      |
|         N=16            |      768.57 ms      |
|         N=32            |      955.05 ms      |
|         N=64            |      569.34 ms      |
|         N=128           |      -              |
 -->

## Interface

The interface is in `include/rnnt.h`. It supports CPU or GPU execution, and you can specify OpenMP parallelism
if running on the CPU, or the CUDA stream if running on the GPU. We took care to ensure that the library does not
preform memory allocation internally, in order to avoid synchronizations and overheads caused by memory allocation.
**Please be carefull if you use the RNNTLoss CPU version, log_softmax should be manually called before the loss function.
(For pytorch binding, this is optionally handled by tensor device.)**

## Compilation

warp-transducer has been tested on Ubuntu 22.04 (WSL2).

First get the code:

```bash
git clone https://github.com/phakhawatchu/warp-transducer.git
cd warp-transducer
```

Install build tools:

```sh
sudo apt update
sudo apt-get update
sudo apt install cmake
sudo apt install build-essential
```

Make sure that warp-transducer is not installed (by simply run)

```sh
pip3 uninstall warprnnt-tensorflow
```

Install latest TensorFlow 2. The following has been tested.

| **Python** | **tensorflow_cpu** | **Working** |
| ---------- | ------------------ | ----------- |
| <= 3.6     | any                | No          |
| 3.7 - 3.9  | <= 2.6.x           | No          |
|            | 2.7.x              | Yes         |
|            | 2.8.x              | Yes         |
|            | >= 2.9.x           | No          |
| >= 3.10    | any                | No          |

```sh
pip install tensorflow
```

There are two choices to build rnnt_loss

-   For CPU, simply run:

```sh
./build_rnnt.sh
```

-   For GPU (when you have CUDA installed at `/usr/local/cuda`). If you have a non standard CUDA install, edit `CUDA_HOME={YOUR_CUDA_PATH}` option.

```sh
export CUDA_HOME=/usr/local/cuda && ./build_rnnt.sh
```

The C library should now be built along with test executables. If CUDA was detected, then `test_gpu` will be built;
`test_cpu` will always be built.

Sometimes, you may have asked to downgrade protobuf version, please run the following:

```sh
pip install protobuf==3.20
```

## Test

To run the tests, make sure the CUDA libraries are in `LD_LIBRARY_PATH` (DYLD_LIBRARY_PATH for OSX).

## Contributing

We welcome improvements from the community, please feel free to submit pull requests.

## Reference

-   [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
-   [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf)
-   [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
-   [Awni implementation of transducer](https://github.com/awni/transducer)
