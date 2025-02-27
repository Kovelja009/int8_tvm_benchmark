import json
import os
import argparse
import time

import torch

import tvm
from tvm import autotvm, testing, auto_scheduler
from tvm.contrib import ndk, rpc, utils
import tvm.contrib.debugger.debug_executor as debug_executor

from utils import quantize

from model_archive import MODEL_ARCHIVE


import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

# Example of usage:
# - GPU cuda target:
'''
    python3 profiling_main.py --model resnet18 --quantize --tuner auto_scheduler --tuning-records resnet18-cuda.json --target cuda --key 1650ti

'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuner", default="autotvm",
                        choices=["autotvm", "auto_scheduler"])
    parser.add_argument("--tuning-records", nargs='?')
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")
    parser.add_argument("--opt-level", default=3, type=int)
    parser.add_argument("--cuda-source", default=False, type=bool)
    args = parser.parse_args()

    assert args.target in ["x86", "arm", "cuda"]

    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    model_info = MODEL_ARCHIVE[args.model]

    model = model_info["model"]()
    input_tensors = model_info["input"]
    model.eval()
    scripted_model = torch.jit.trace(model, input_tensors).eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    mod, params = tvm.relay.frontend.from_pytorch(
        scripted_model, input_infos)

    if args.quantize:
        with open("quantization_scheme.json", "r") as read_file:
            quantization_scheme = json.load(read_file)
        print("Quantization params: ", quantization_scheme)
        mod = quantize(mod, params, False, **quantization_scheme)

    if args.target == "x86":
        target = "llvm -mcpu=cascadelake"
    elif args.target == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
    elif args.target == "cuda":
        target = "cuda"


    def relay_build(use_auto_scheduler):
        with tvm.transform.PassContext(opt_level=args.opt_level, config={"relay.backend.use_auto_scheduler": use_auto_scheduler}):
            return tvm.relay.build(mod, target=target, params=params)

    if args.tuning_records is None:
        lib = relay_build(False)
    elif args.tuner == "autotvm":
        with autotvm.apply_history_best(args.tuning_records):
            lib = relay_build(False)
    elif args.tuner == "auto_scheduler":
        with auto_scheduler.ApplyHistoryBest(args.tuning_records):
            lib = relay_build(True)


    if args.cuda_source:
        gpu_source_code = lib.get_lib().imported_modules[0].get_source()
        
        # save the source code to a file
        with open("cuda_code.cu", "w") as f:
            f.write(gpu_source_code)



    if args.target == "x86":
        ctx = tvm.device(str(target), 0)
        m = debug_executor.create(
            lib.get_graph_json(), lib.get_lib(), ctx)
    elif args.target == "arm":
        libname = "model.so"
        temp = utils.tempdir()
        libpath = temp.relpath(libname)
        lib.export_library(libpath, ndk.create_shared)
        remote = rpc.connect_tracker(args.host, args.port).request(args.key)
        remote.upload(libpath)
        rlib = remote.load_module(libname)
        ctx = remote.cpu(0)
        m = debug_executor.create(lib.get_graph_json(), rlib, ctx)
    elif args.target == "cuda":
        ctx = tvm.cuda(0)
        m = debug_executor.create(
            lib.get_graph_json(), lib.get_lib(), ctx)



    for input_info, input_tensor in zip(input_infos, input_tensors):
        m.set_input(
            input_info[0],
            tvm.nd.array(input_tensor.cpu().numpy(), ctx)
        )
    m.set_input(**lib.get_params())

    print(m.benchmark(ctx, number=1, repeat=600))

    report = m.profile()
    print(report)

    with open("%s.csv" % time.strftime("%Y%m%d-%H%M%S"), "w") as f:
        f.write(report.csv())

    with torch.no_grad():
        outputs = model(*input_tensors)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for i in range(len(outputs)):
        testing.assert_allclose(
            m.get_output(i).numpy(), outputs[i].cpu().numpy(), rtol=1e-4)
