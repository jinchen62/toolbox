import argparse
import subprocess
import numpy as np
import re

matmul_transpose_b = False
matmul_transpose_a = True

def generate_matmul_func(m, n, k):
    """Generates the MxNxK matrix multiplication function in MLIR."""
    global matmul_transpose_a
    if matmul_transpose_a:
        matmul_function = (\
        f"func.func @matmul(%lhs: tensor<{k}x{m}xf16>, %rhs: tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16> {{\n"
        f"  %c0 = arith.constant 0.0 : f16\n"
        f"  %init = tensor.empty() : tensor<{m}x{n}xf16>\n"
        f"  %inital_result = linalg.fill ins(%c0 : f16) outs(%init : tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  %result = linalg.matmul_transpose_a ins(%lhs, %rhs: tensor<{k}x{m}xf16>, tensor<{k}x{n}xf16>)\n"
        f"             outs(%inital_result: tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  return %result : tensor<{m}x{n}xf16>\n"
        f"}}\n")
    elif matmul_transpose_b:
        matmul_function = (\
        f"func.func @matmul(%lhs: tensor<{m}x{k}xf16>, %rhs: tensor<{n}x{k}xf16>) -> tensor<{m}x{n}xf16> {{\n"
        f"  %c0 = arith.constant 0.0 : f16\n"
        f"  %init = tensor.empty() : tensor<{m}x{n}xf16>\n"
        f"  %inital_result = linalg.fill ins(%c0 : f16) outs(%init : tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<{m}x{k}xf16>, tensor<{n}x{k}xf16>)\n"
        f"             outs(%inital_result: tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  return %result : tensor<{m}x{n}xf16>\n"
        f"}}\n")
    else:
        matmul_function = (\
        f"func.func @matmul(%lhs: tensor<{m}x{k}xf16>, %rhs: tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16> {{\n"
        f"  %c0 = arith.constant 0.0 : f16\n"
        f"  %init = tensor.empty() : tensor<{m}x{n}xf16>\n"
        f"  %inital_result = linalg.fill ins(%c0 : f16) outs(%init : tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  %result = linalg.matmul ins(%lhs, %rhs: tensor<{m}x{k}xf16>, tensor<{k}x{n}xf16>)\n"
        f"             outs(%inital_result: tensor<{m}x{n}xf16>) -> tensor<{m}x{n}xf16>\n"
        f"  return %result : tensor<{m}x{n}xf16>\n"
        f"}}\n")
    return matmul_function

def execute_command(command, output_file=''):
    """Executes the given command and logs the output to the given file."""

    print('Executing command: ', command)
    print(' '.join(command))
    out = None
    err = None
    if output_file != '':
        with open(output_file, 'w') as f:
            process = subprocess.Popen(command, stderr=f)
            process.wait()
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        out, err = process.communicate()
        process.wait()
    return out, err

def generate_mlir(args):
    matmul_str = generate_matmul_func(args.m, args.n, args.k)
    fname = f'matmul_m{args.m}_n{args.n}_k{args.k}.mlir'
    with open(fname, 'w') as f:
        f.write(matmul_str)
    return fname


def compile(args):
    command = ['../iree-build/tools/iree-compile',
           '--iree-input-type=tm_tensor',
           '--iree-vm-bytecode-module-output-format=flatbuffer-binary',
           '--iree-hal-target-backends=rocm',
           '--iree-rocm-target-chip=gfx1100',
           '--iree-rocm-link-bc=true',
           '--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode',
           f'{args.fname}',
           '-o', 'matmul.vmfb']
    if args.transform_dialect:
        command += ['--iree-codegen-llvmgpu-use-transform-dialect=matmul_spec.mlir',
               '--iree-codegen-llvmgpu-enable-transform-dialect-jit=false']
    if args.dump:
        command += ['-mlir-print-ir-after-all',
                    '-mlir-disable-threading',
                    '--iree-hal-dump-executable-binaries-to=/home/harsh/iree/tmp']
    execute_command(command, 'mlirdump.txt')

def validate(args):
    global matmul_transpose_a, matmul_transpose_b
    if matmul_transpose_a:
        lhs = 0.005 * np.random.rand(int(args.k), int(args.m)).astype('float16')
        rhs = 0.005 * np.random.rand(int(args.k), int(args.n)).astype('float16')
        output = np.matmul(np.transpose(lhs), rhs)
    elif matmul_transpose_b:
        lhs = 0.005 * np.random.rand(int(args.m), int(args.k)).astype('float16')
        rhs = 0.005 * np.random.rand(int(args.n), int(args.k)).astype('float16')
        output = np.matmul(lhs, np.transpose(rhs))
    else:
        lhs = 0.005 * np.random.rand(int(args.m), int(args.k)).astype('float16')
        rhs = 0.005 * np.random.rand(int(args.k), int(args.n)).astype('float16')
        output = np.matmul(lhs, rhs)
    lhs_filename = f'lhs_m{args.m}_n{args.n}_k{args.k}.npy'
    with open(lhs_filename, 'wb') as f:
        np.save(f, lhs)
    rhs_filename = f'rhs_m{args.m}_n{args.n}_k{args.k}.npy'
    with open(rhs_filename, 'wb') as f:
        np.save(f, rhs)
    output_filename = f'output_m{args.m}_n{args.n}_k{args.k}.npy'
    with open(output_filename, 'wb') as f:
        np.save(f, output)
    command = ['../iree-build/tools/iree-run-module',
           '--device=rocm',
           '--module=matmul.vmfb',
           '--function="matmul"',
           f'--input=@{lhs_filename}',
           f'--input=@{rhs_filename}',
           f'--expected_output=@{output_filename}']
    out, err = execute_command(command)
    output = out.decode('utf-8')
    print(output)
    
def benchmark(args):
    global matmul_transpose_a, matmul_transpose_b
    if matmul_transpose_a:
        command = ['../iree-build/tools/iree-benchmark-module',
               '--module=matmul.vmfb',
               '--function=matmul',
               f'--input="{args.k}x{args.m}xf16"',
               f'--input="{args.k}x{args.n}xf16"',
               '--device=rocm',
               '--batch_size=100']
    elif matmul_transpose_b:
        command = ['../iree-build/tools/iree-benchmark-module',
               '--module=matmul.vmfb',
               '--function=matmul',
               f'--input="{args.m}x{args.k}xf16"',
               f'--input="{args.n}x{args.k}xf16"',
               '--device=rocm',
               '--batch_size=100']
    else:
        command = ['../iree-build/tools/iree-benchmark-module',
               '--module=matmul.vmfb',
               '--function=matmul',
               f'--input="{args.m}x{args.k}xf16"',
               f'--input="{args.k}x{args.n}xf16"',
               '--device=rocm',
               '--batch_size=100']
    out, err = execute_command(command)
    output = out.decode('utf-8')
    print(output)
    time = float(re.findall(r"[-+]?(?:\d*\.*\d+)", output.split('\n')[3])[0])
    flops = (2 * int(args.m) * int(args.n) * int(args.k) / (time * 1e-3) ) / (1e12)
    print("Throughput (TFLOPS/s) = ", flops)


parser = argparse.ArgumentParser(description='Matmul ROCM runner')

parser.add_argument('-i', '--input_file', help='The input mlir file.')
parser.add_argument('-s', '--spec_file', help='The spec file.')
parser.add_argument('-m', help='matrix M shape.')
parser.add_argument('-n', help='matrix N shape.')
parser.add_argument('-k', help='matrix K shape.')
parser.add_argument('-d', '--dump', action='store_true', help='Dump the ir after all.')
parser.add_argument('-c', '--compile', action='store_true', help='Compile the program.')
parser.add_argument('-r', '--run', action='store_true', help='Run the program.')
parser.add_argument('-b', '--benchmark', action='store_true', help='Benchmark the program.')
parser.add_argument('-t', '--transform_dialect', action='store_true', help='Use td script')

args = parser.parse_args()

fname = generate_mlir(args)
args.fname = fname

if args.compile:
    print('Compiling the program...')
    compile(args)
if args.run:
    print('Running the program...')
    validate(args)
if args.benchmark:
    print('Benchmarking the program...')
    benchmark(args)
