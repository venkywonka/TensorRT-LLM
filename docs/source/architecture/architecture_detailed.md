(detailed-trtllm-architecture)=
# Detailed TRT-LLM Architecture

The following diagram illustrates the detailed architecture of TensorRT-LLM, including the offline preparation phase and the online runtime phase, highlighting model conversion, PEFT integration, and low-latency optimizations.

```mermaid
graph TD
    %% I. Offline/Preparation Phase
    subgraph Offline/Preparation Phase
        direction LR

        %% A. User Inputs
        subgraph User Inputs
            direction TB
            HF_CKPT[("Hugging Face Pretrained Model Checkpoint (Weights + Config)")]
            HF_LORA_CKPT[("Hugging Face LoRA Adapter Checkpoints (optional)")]
        end

        %% B. PyTorch Layer
        subgraph PyTorch Layer (`tensorrt_llm._torch.models`)
            direction TB
            PYTORCH_MODELS["Model Definitions (e.g., Llama, GPT)<br/>Specialized for TRT-LLM TP/PP"]
            AUTO_MODEL["`AutoModelForCausalLM`"]
        end

        %% C. Conversion & Preprocessing Tools
        subgraph Conversion & Preprocessing Tools (`examples/`)
            direction TB
            CONVERT_CKPT["`convert_checkpoint.py`<br/>(model-specific, e.g., for Llama, Eagle)"]
            HF_LORA_CONVERT["`hf_lora_convert.py`"]
        end

        %% D. TRT-LLM Formatted Checkpoint & Processed LoRA
        TRTLLM_CKPT[("TRT-LLM Formatted Checkpoint<br/>(config.json + sharded weights)")]
        PROCESSED_LORA[("Processed LoRA Weights & Config<br/>(.npy files)")]

        %% E. trtllm-build
        subgraph `trtllm-build` (Command-Line Tool)
            direction TB
            TRTLLM_BUILD_CMD["`trtllm-build` CLI"]
            BUILDER_CONFIG[("Builder Config<br/>(precision, plugins, batch_size, LoRA config, Speculative Mode)")]

            subgraph Internals
                direction TB
                BUILDER_API["TensorRT-LLM Builder API<br/>(`tensorrt_llm.Builder`)"]
                TRT_COMPILER["TensorRT Compiler<br/>(kernel selection, fusions, CUDA graph)"]
                TRTLLM_PLUGINS["TensorRT-LLM Plugins<br/>(FlashAttention, GEMM, LoRA, etc.)"]
            end
            BUILDER_API --> TRT_COMPILER
            BUILDER_API --> TRTLLM_PLUGINS
        end

        OPTIMIZED_ENGINE[("Optimized TensorRT Engine(s)<br/>(may include embedded LoRA & Speculative Decoding)")]

        %% Flows for Offline Phase
        HF_CKPT -->|Consumed by| CONVERT_CKPT
        PYTORCH_MODELS -->|Referenced by| CONVERT_CKPT
        AUTO_MODEL -->|Used by| CONVERT_CKPT
        CONVERT_CKPT -->|Outputs| TRTLLM_CKPT

        HF_LORA_CKPT -->|Consumed by| HF_LORA_CONVERT
        HF_LORA_CONVERT -->|Outputs| PROCESSED_LORA

        TRTLLM_CKPT -->|Consumed by| TRTLLM_BUILD_CMD
        PROCESSED_LORA -->|Consumed by (optional)| TRTLLM_BUILD_CMD
        BUILDER_CONFIG -->|Input to| TRTLLM_BUILD_CMD
        TRTLLM_BUILD_CMD -- uses --> BUILDER_API
        TRTLLM_BUILD_CMD -->|Outputs| OPTIMIZED_ENGINE
    end

    %% II. Online/Runtime Phase
    subgraph Online/Runtime Phase
        direction LR

        %% A. Client Application
        subgraph Client Application (Python or C++)
            direction TB
            CLIENT_APP["Client"]
            REQUEST[("Request<br/>(Input Data, Sampling Params, LoRA UID, Speculative Config)")]
            CLIENT_APP -- Constructs --> REQUEST
            PYTHON_BINDINGS["Python Bindings"]
            CPP_API["C++ API"]
        end

        %% B. TensorRT-LLM C++ Backend
        subgraph TensorRT-LLM C++ Backend
            direction TB
            EXECUTOR_API["Executor API (C++)"]
            CPP_RUNTIME["C++ Runtime<br/>(`TllmRuntime`, `GptDecoder`)"]

            subgraph Supporting Components
                direction RL
                LORA_CACHE[("LoRA Cache<br/>(CPU/GPU, from engine)")]
                KV_CACHE[("K/V Cache<br/>(Paged K/V)")]
            end

            EXECUTOR_API -- Manages --> LORA_CACHE
            CPP_RUNTIME -- Manages --> KV_CACHE
            EXECUTOR_API -- Forwards to --> CPP_RUNTIME
        end

        %% C. TensorRT Engine(s)
        RUNTIME_ENGINE[("TensorRT Engine(s)<br/>(Loaded by C++ Runtime)")]

        %% Flows for Online Phase
        CLIENT_APP -- via --> PYTHON_BINDINGS
        CLIENT_APP -- via --> CPP_API
        REQUEST --> PYTHON_BINDINGS
        REQUEST --> CPP_API
        PYTHON_BINDINGS --> EXECUTOR_API
        CPP_API --> EXECUTOR_API

        EXECUTOR_API -- Receives --> REQUEST
        EXECUTOR_API -- Orchestrates LoRA & Speculative Decoding --> RUNTIME_ENGINE
        CPP_RUNTIME -- Loads/Manages & Executes --> RUNTIME_ENGINE

        RESPONSE[("Response")]
        EXECUTOR_API -- Returns --> RESPONSE
        RESPONSE --> CLIENT_APP

    end

    %% Connections between phases
    OPTIMIZED_ENGINE -->|Loaded by| CPP_RUNTIME

    %% Styling (optional, for better readability)
    classDef userInputs fill:#DCDCDC,stroke:#333,stroke-width:2px;
    classDef pytorchLayer fill:#C9DAF8,stroke:#333,stroke-width:2px;
    classDef conversionTools fill:#FCE5CD,stroke:#333,stroke-width:2px;
    classDef trtllmObjects fill:#E6B9B8,stroke:#333,stroke-width:2px;
    classDef buildTool fill:#D9EAD3,stroke:#333,stroke-width:2px;
    classDef clientApp fill:#FFF2CC,stroke:#333,stroke-width:2px;
    classDef cppBackend fill:#B4A7D6,stroke:#333,stroke-width:2px;
    classDef runtimeEngine fill:#A2C4C9,stroke:#333,stroke-width:2px;

    class HF_CKPT,HF_LORA_CKPT userInputs;
    class PYTORCH_MODELS,AUTO_MODEL pytorchLayer;
    class CONVERT_CKPT,HF_LORA_CONVERT conversionTools;
    class TRTLLM_CKPT,PROCESSED_LORA,OPTIMIZED_ENGINE,RUNTIME_ENGINE trtllmObjects;
    class TRTLLM_BUILD_CMD,BUILDER_CONFIG,BUILDER_API,TRT_COMPILER,TRTLLM_PLUGINS buildTool;
    class CLIENT_APP,REQUEST,RESPONSE,PYTHON_BINDINGS,CPP_API clientApp;
    class EXECUTOR_API,CPP_RUNTIME,LORA_CACHE,KV_CACHE cppBackend;
```

---

### Chart Explanation

This diagram outlines the end-to-end workflow of TensorRT-LLM, from model preparation to inference. It's divided into two main phases:

**I. Offline/Preparation Phase:**

This phase covers all the steps necessary to convert a pretrained Large Language Model (LLM) into an optimized TensorRT engine.

1.  **User Inputs:**
    *   **Hugging Face (HF) Pretrained Model Checkpoint:** Standard model weights and configuration from sources like the Hugging Face Hub.
    *   **HF LoRA Adapter Checkpoints (Optional):** If using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, these are the adapter weights.

2.  **PyTorch Layer (`tensorrt_llm._torch.models`):**
    *   TensorRT-LLM utilizes its own PyTorch model definitions (e.g., `LlamaModel`, `GptModel`). These are designed to be compatible with TensorRT conversion and often include logic for distributed training/inference concepts like Tensor Parallelism (TP) and Pipeline Parallelism (PP).
    *   `AutoModelForCausalLM` helps in loading the appropriate model class based on the input checkpoint's configuration.

3.  **Conversion & Preprocessing Tools (`examples/`):**
    *   **`convert_checkpoint.py`:** These are model-specific scripts (e.g., found in `examples/models/core/llama/`) that take the original HF checkpoint and the corresponding `tensorrt_llm._torch` model definition.
        *   They process the weights, handle sharding for TP/PP, and can apply initial quantization (like weight-only, SmoothQuant, FP8) to the base model.
        *   For low-latency optimizations like **Eagle speculative decoding**, specialized versions of these scripts (e.g., `examples/eagle/convert_checkpoint.py`) merge the draft model weights with the main model weights.
        *   The output is a **TRT-LLM Formatted Checkpoint**, which includes a TensorRT-LLM specific `config.json` and sharded model weights (often in `.safetensors` format).
    *   **`hf_lora_convert.py`:** This script specifically converts HF LoRA adapter checkpoints into a format suitable for `trtllm-build` (typically NumPy `.npy` files for weights and configuration).

4.  **`trtllm-build` (Command-Line Tool):**
    *   This is the core tool for compiling the model into a TensorRT engine.
    *   It consumes the **TRT-LLM Formatted Checkpoint**.
    *   Optionally, it takes the **Processed LoRA Weights & Config** if LoRA is to be embedded in the engine.
    *   **Builder Config** provides crucial parameters for compilation, such as:
        *   Precision settings (e.g., FP16, BF16, FP8).
        *   Plugin selection (e.g., GEMM plugin, GPT attention plugin).
        *   Maximum batch size, input/output lengths.
        *   LoRA plugin configuration if LoRA is used.
        *   **Speculative Decoding Mode** (e.g., `'eagle'` to build an engine with integrated Eagle drafter).
    *   **Internals:**
        *   **TensorRT-LLM Builder API (`tensorrt_llm.Builder`):** The Python API used by `trtllm-build` to define the network for TensorRT.
        *   **TensorRT Compiler:** Performs graph optimizations, kernel selection, layer fusions, and generates a CUDA graph.
        *   **TensorRT-LLM Plugins:** Custom, highly optimized kernels for operations like FlashAttention, specific GEMM variants, LoRA layers, etc., which TensorRT might not generate optimally by default.
    *   The output is one or more **Optimized TensorRT Engine(s)**. These engines are highly efficient versions of the LLM, ready for inference. They can have LoRA capabilities and speculative decoding logic (like the Eagle drafter) compiled directly into them.

**II. Online/Runtime Phase:**

This phase describes how the optimized TensorRT engine is used for inference.

1.  **Client Application (Python or C++):**
    *   The application that needs to run LLM inference.
    *   It constructs a **`Request`** object containing:
        *   Input data (e.g., tokenized text).
        *   Sampling parameters (e.g., temperature, top-k, top-p).
        *   **LoRA Task UID (User ID):** If LoRA is used, this identifier tells the backend which LoRA adapter (that was built into the engine) to activate for this request.
        *   **Speculative Decoding Configuration:** Parameters related to speculative decoding if used.
    *   Clients can interact with the C++ backend via **Python Bindings** (for Python applications) or directly using the **C++ API**.

2.  **TensorRT-LLM C++ Backend:** This is the high-performance inference server.
    *   **Executor API (C++):**
        *   The primary interface for the backend. It receives `Request` objects.
        *   Manages a request queue and performs **in-flight batching** to optimize GPU utilization by batching multiple requests together.
        *   **Orchestrates LoRA:** Based on the LoRA Task UID in the request, it configures the loaded engine to use the appropriate LoRA weights. It utilizes a **`LoRA Cache`** (which can be on CPU or GPU) to efficiently manage and switch between different LoRA adapters that were compiled into the engine.
        *   **Orchestrates Speculative Decoding:** If speculative decoding (like Eagle) is part of the compiled engine, the Executor manages its execution. For other speculative methods that might involve separate draft models (though less common with integrated approaches like Eagle), it would coordinate their execution.
        *   Forwards processed batches to the C++ Runtime for execution on the GPU.
        *   Receives results from the C++ Runtime and constructs a **`Response`** to send back to the client.
    *   **C++ Runtime (`TllmRuntime`, `GptDecoder`):**
        *   Loads and manages the **TensorRT Engine(s)**.
        *   Executes the engine(s) on the GPU(s).
        *   Manages the **K/V Cache** (Key-Value Cache for attention layers), often using a Paged K/V cache for efficient memory management.
        *   Handles **Multi-GPU/Multi-Node** execution details, leveraging Tensor Parallelism and Pipeline Parallelism through NCCL plugins for communication.

3.  **TensorRT Engine(s):**
    *   The actual compiled model that runs on the GPU. As noted, these engines can be pre-compiled with specific LoRA adapters and speculative decoding logic.

**Model Organization & Backends Summary:**

*   **PyTorch's Role:** Primarily in the **Offline Phase**. TensorRT-LLM provides PyTorch modules (`tensorrt_llm._torch.models`) that are used as the starting point for defining the model structure. Standard PyTorch checkpoints are converted using these definitions. There isn't a "PyTorch inference backend" in TRT-LLM; instead, PyTorch is a crucial part of the model preparation pipeline before handing off to the C++ inference backend.
*   **C++ Backend:** This is the core inference engine of TensorRT-LLM, designed for high performance and efficiency. It comprises the Executor for request management and advanced feature orchestration, and the C++ Runtime for direct engine execution and hardware interaction.
*   **Model Structure:** Models are defined in PyTorch, converted to a TRT-LLM specific format, and then compiled into TensorRT engines. The structure within the engine is highly optimized and may not directly mirror the original PyTorch layer-by-layer structure due to fusions and other optimizations.
*   **PEFT (LoRA/DoRA):** Handled by converting LoRA adapters and building them into the engine. The Executor then selects the appropriate adapter at runtime.
*   **Low-Latency Optimizations (Eagle, etc.):** Techniques like Eagle are integrated by compiling their logic directly into the TensorRT engine. The Executor then utilizes these specialized engines.

This architecture allows TensorRT-LLM to leverage PyTorch's flexibility for model development and training, while providing a highly optimized C++ backend for production inference with support for advanced features like PEFT and various low-latency techniques.
