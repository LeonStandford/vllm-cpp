# 🧰 Technical prerequisities

You can build and run it on any platform, with minor changes, assuming you have a NVIDIA GPU. You might need to adjust some paths, like CUDA or GCC in [c_cpp_propertiesjson](../.vscode/c_cpp_properties.json) or NVCC in [CMakeLists.txt](../CMakeLists.txt)

I suggest you to fork this repo and make the necessary adjustments so it works on your machine and then create a pull request to [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) and upstream your changes for benefit of another readers

The exact setup on which I develop and test it:
- Linux (6.19.8 x64_64)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (13.1)
- C++ 17
- [GCC](https://gcc.gnu.org/) (15.2.1)
- The only external dependency you will pull in is JSON parser [nlohmann/json](https://github.com/nlohmann/json) 3.12.0, which is a single header file [include/json.hpp](../include/json.hpp)
- AMD CPU (Ryzen 7 9800X3D)
- NVIDIA GPU (RTX 5090)
- I used [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) from Hugging Face (commit hash 898999bd25b40516fce5a5b8f0948f4c81c650bc), you need just `model.safetensors` file from this repository

Install the dependencies and run the program with `./test.sh` - it will build it and immediately execute it

It also runs on AMD GPUs through ROCm/HIP. Pass `-DUSE_HIP=ON` to CMake and it builds with hipcc against hipBLAS instead of nvcc and cuBLAS; the CUDA sources are reused as-is through a thin `src/cuda_to_hip.h` compatibility header. Pick your GPU's architecture with `-DCMAKE_HIP_ARCHITECTURES` (for example `gfx90a` for MI200, `gfx1100` for RDNA3, `gfx1201` for RDNA4) - it is not hardcoded, so set it to match your card:

```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm -G Ninja
cmake --build build
```

The `-DCMAKE_PREFIX_PATH=/opt/rocm` lets CMake find the hip and hipBLAS packages; drop it if `/opt/rocm/bin` is already on your `PATH`, or change it if ROCm lives elsewhere. I tested the AMD path on gfx90a, gfx1100, and gfx1201. The default build (no `-DUSE_HIP`) is unchanged and still targets NVIDIA through CUDA.

If you fail to build or run it and your AI of choice won't be able to help, please open an Issue on GitHub - I will try to help. Make sure to provide all useful context

---

[← Intro: LLM, vLLM, models, inference servers](01-intro-llm-vllm-models-inference-servers.md) · [🏠 Index](../README.md) · [Safetensors and your model →](03-safetensors-and-your-model.md)
