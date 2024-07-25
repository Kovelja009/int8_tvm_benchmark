## Tuning

First we need to start RPC tracker. The tracker is required during the whole tuning process, so we need to open a new terminal for this command:
```
python3 -m tvm.exec.rpc_tracker --host=127.0.0.1 --port=9190
```
Then in another terminal, we can start the tuning script
```
    python3 tuning_main.py --model resnet18 --quantize --target x86 --key intel_thinkpad
```

Output should look something like this:
```
Extract tasks...
Tuning...
[Task  1/12]  Current/Best:  541.83/3570.66 GFLOPS | Progress: (960/2000) | 1001.31 s Done.
[Task  2/12]  Current/Best:    0.56/ 803.33 GFLOPS | Progress: (704/2000) | 608.08 s Done.
[Task  3/12]  Current/Best:  103.69/1141.25 GFLOPS | Progress: (768/2000) | 702.13 s Done.
[Task  4/12]  Current/Best: 2905.03/3925.15 GFLOPS | Progress: (864/2000) | 745.94 sterminate called without an active exception
[Task  4/12]  Current/Best: 2789.36/3925.15 GFLOPS | Progress: (1056/2000) | 929.40 s Done.
[Task  5/12]  Current/Best:   89.06/1076.24 GFLOPS | Progress: (704/2000) | 601.73 s Done.
[Task  6/12]  Current/Best:   40.39/2129.02 GFLOPS | Progress: (1088/2000) | 1125.76 s Done.
[Task  7/12]  Current/Best: 4090.53/5007.02 GFLOPS | Progress: (800/2000) | 903.90 s Done.
[Task  8/12]  Current/Best:    4.78/1272.28 GFLOPS | Progress: (768/2000) | 749.14 s Done.
[Task  9/12]  Current/Best: 1391.45/2325.08 GFLOPS | Progress: (992/2000) | 1084.87 s Done.
[Task 10/12]  Current/Best: 1995.44/2383.59 GFLOPS | Progress: (864/2000) | 862.60 s Done.
[Task 11/12]  Current/Best: 4093.94/4899.80 GFLOPS | Progress: (224/2000) | 240.92 sterminate called without an active exception
[Task 11/12]  Current/Best: 3487.98/4909.91 GFLOPS | Progress: (480/2000) | 534.96 sterminate called without an active exception
[Task 11/12]  Current/Best: 4636.84/4912.17 GFLOPS | Progress: (1184/2000) | 1381.16 sterminate called without an active exception
[Task 11/12]  Current/Best:   50.12/4912.17 GFLOPS | Progress: (1344/2000) | 1602.81 s Done.
[Task 12/12]  Current/Best: 3581.31/4286.30 GFLOPS | Progress: (736/2000) | 943.52 s Done.
Compile...
Evaluate inference time cost...
Mean inference time (std dev): 1.07 ms (0.05 ms)
```
