## Tuning
We can start the tuning script:
- Examples:
    - for `CPU`:
        ```
        python3 tuning_main.py --model resnet18 --quantize --tuner auto_scheduler --target x86 --key intel_cpu
        ```
    - for `GPU`:
        ```
            python3 tuning_main.py --model resnet18 --quantize --tuner auto_scheduler --target cuda --key 1650ti
        ```
## Profiling
We can start the profiling script:
- Examples:
```
python3 profiling_main.py --model resnet18 --quantize --tuner auto_scheduler --tuning-records resnet18-cuda.json --target cuda --key 1650ti
```
In `quantization_scheme.json` you can set quantization parameters. Supported are `bool`, `int8`, `int16` and `int32`. Here are default parameters:
```
{
       "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "calibrate_mode": "global_scale",
        "global_scale": 8.0,
        "weight_scale": "power2",
        "skip_dense_layer": True,
        "skip_conv_layers": [0],
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
        "rounding": "UPWARD",
        "calibrate_chunk_by": -1,
        "partition_conversions": "disabled",
}
```

