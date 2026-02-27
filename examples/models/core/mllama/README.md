# MLLaMA (llama-3.2 Vision model)

> [!WARNING]
> The `convert_checkpoint.py` / `trtllm-build` / `run.py` workflow described
> below is **legacy** and will not receive new features. New projects should use
> [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
> or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html) instead.

MLLaMA is a multimodal model, and reuse the multimodal modules in [examples/models/core/multimodal](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
