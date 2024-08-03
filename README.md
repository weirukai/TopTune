# TopTune: Tailored Optimization for Categorical and Continuous Knobs Towards Accelerated and Improved Database Performance Tuning
Source code of the tuning model TopTune.

> **TopTune: Tailored Optimization for Categorical and Continuous Knobs Towards Accelerated and Improved Database Performance Tuning**  
Rukai Wei, Yu Liu, Yufeng Hou, Heng Cui, Yongqiang Zhang, Ke Zhou.

---
## Installation
1. Preparations: Python == 3.7

2. Install packages

   ```shell
   pip install -r requirements.txt

## Benchmark Preparation

1. https://github.com/akopytov/sysbench [SYSBENCH]
2. https://github.com/petergeoghegan/benchmarksql [TPCC]
3. https://github.com/winkyao/join-order-benchmark [JOB]

## Run
To start a training session, please run:
  ```python
  python main.py
  ```
## Acknowledgment

The structure of this code is largely based on [UniTune](https://github.com/Blairruc-pku/UniTune) and [OpenBox](https://github.com/PKU-DAIR/open-box). Thank them for their wonderful works.
