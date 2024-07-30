import os
import argparse

import torch
import torch.utils.dlpack

import tvm
from tvm import autotvm, auto_scheduler
import tvm.relay

from utils import quantize, tune_network, tune_network_auto_scheduler
from model_archive import MODEL_ARCHIVE


# TODO: Add support for cuda target
# Example of usage:
# First we need to start RPC tracker. The tracker is required during the whole tuning process, so we need to open a new terminal for this command:
'''
    python3 -m tvm.exec.rpc_tracker --host=127.0.0.1 --port=9190
'''
# Then in another terminal, we can start the tuning script for:
# - CPU x86 target:
'''
    python3 tuning_main.py --model resnet18 --quantize --tuner auto_scheduler --target x86 --key 1650ti
'''
# - GPU cuda target:
'''
    python3 tuning_main.py --model resnet18 --quantize --tuner auto_scheduler --target cuda --key 1650ti
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuning-records", default="resnet18.json")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--tuner", default="autotvm",
                        choices=["autotvm", "auto_scheduler"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86", choices=["x86", "arm", "cuda"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="intel_thinkpad")
    args = parser.parse_args()

    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    model_info = MODEL_ARCHIVE[args.model]

    log_file = "%s-%s.log" % (args.model, args.target)

    print("Loading model from PyTorch")
    model = model_info["model"]()
    input_tensors = model_info["input"]
    scripted_model = torch.jit.trace(model, input_tensors).eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    mod, params = tvm.relay.frontend.from_pytorch(
        scripted_model, input_infos)

    if args.quantize:
        print("Quantizing model")
        mod = quantize(mod, params, False)

    if args.target == "x86":
        print("Using x86 target")
        target = "llvm -mcpu=cascadelake"
        if args.tuner == "autotvm":
            print("Using autotvm tuner")
            measure_option = autotvm.measure_option(
                builder="local", runner="local"
            )
        elif args.tuner == "auto_scheduler":
            print("Using auto_scheduler tuner")
            builder = auto_scheduler.LocalBuilder()
            runner = auto_scheduler.LocalRunner(
                repeat=10, enable_cpu_cache_flush=True
            )
    elif args.target == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        if args.tuner == "autotvm":
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="ndk", timeout=60),
                runner=autotvm.RPCRunner(
                    args.key, args.host, args.port)
            )
        elif args.tuner == "auto_scheduler":
            builder = auto_scheduler.LocalBuilder(build_func="ndk", timeout=60)
            runner = auto_scheduler.RPCRunner(
                args.key,
                host=args.host,
                port=args.port,
                repeat=5,
                min_repeat_ms=200,
                enable_cpu_cache_flush=True,
            )
    elif args.target == "cuda":
        target = tvm.target.Target("cuda")
        if args.tuner == "autotvm":
            print("Autotvm not supported for cuda target yet :(")
            exit(0)
        elif args.tuner == "auto_scheduler":
                measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    if args.tuner == "autotvm":
        tuning_option = {
            "n_trial": 1000,
            "early_stopping": 500,
            "measure_option": measure_option,
            "tuning_records": log_file,
        }
        print("Tuning network with autotvm...")
        tune_network(mod, params, target, tuning_option)
    elif args.tuner == "auto_scheduler":
        if args.target == "cuda":
            tuning_option = auto_scheduler.TuningOptions(
                num_measure_trials=15,
                runner=measure_ctx.runner,
                measure_callbacks=[
                    auto_scheduler.RecordToFile(log_file)],
            )
        else:
            tuning_option = auto_scheduler.TuningOptions(
                num_measure_trials=15,
                builder=builder,
                runner=runner,
                measure_callbacks=[
                    auto_scheduler.RecordToFile(log_file)],
            )
        print("Tuning network with auto_sheduler...")
        tune_network_auto_scheduler(mod, params, target, tuning_option)