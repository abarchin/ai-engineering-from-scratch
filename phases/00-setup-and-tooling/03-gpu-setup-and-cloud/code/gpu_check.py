import time
import sys

def check_gpu() -> bool:
    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.mps.is_available()}")

    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("\nNo GPU detected. That's fine for most lessons.")
        print("For GPU-heavy lessons, use Google Colab (free).")
        return False

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

    return True

def run_gpu_benchmark(device: str = "mps"):
    print("\n=== CPU vs GPU Benchmark ===\n")
    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    a_gpu = a.to(device)
    b_gpu = b.to(device)
    sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize
    sync()

    # Warmup — first kernel call compiles Metal shaders / loads CUDA context
    _ = a_gpu @ b_gpu
    sync()

    start = time.time()
    for _ in range(10):
        _ = a_gpu @ b_gpu
    sync()
    gpu_time = (time.time() - start) / 10
    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

    if device == "cuda":
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
    else:
        import subprocess
        total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        vram_bytes = int(total_ram * 0.7)

    vram_gb = vram_bytes / 1e9
    params_billions = (vram_bytes / 2) / 1e9
    print(f"\nUsable GPU memory: ~{vram_gb:.1f} GB")
    print(f"Estimated max model size (fp16): ~{params_billions:.0f}B parameters")


if __name__ == "__main__":
    try:
        import torch
        if check_gpu():
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            run_gpu_benchmark(device)
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
