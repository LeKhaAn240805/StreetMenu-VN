"""Serving presets used by `smv models gen` to write configs/models/<key>.yaml.

Keys and tiers come from ref/models_candidates.yaml; this table adds what vLLM needs (verified on vLLM 0.32
architecture list and on each model card) and the extra models agreed in Step 0 (Vietnamese, quantised).
Revisions are filled from configs/models/VERIFIED.csv (`smv models verify`)."""
from __future__ import annotations

from typing import Any

QWEN_MM = [{"size": {"longest_edge": 2097152, "shortest_edge": 65536}},
           {"size": {"longest_edge": 1048576, "shortest_edge": 65536}}]
QWEN2_MM = [{"max_pixels": 1605632}, {"max_pixels": 802816}]
INTERNVL_MM = [{"max_dynamic_patch": 6}, {"max_dynamic_patch": 4}]
GLMV_MM = [{"size": {"longest_edge": 2097152, "shortest_edge": 12544}},
           {"size": {"longest_edge": 1048576, "shortest_edge": 12544}}]
THINK_FLAG = {"supported": True, "default_on": False,
              "off": {"chat_template_kwargs": {"enable_thinking": False}},
              "on": {"chat_template_kwargs": {"enable_thinking": True}}}
THINK_DEFAULT_ON = {**THINK_FLAG, "default_on": True}

# family presets -------------------------------------------------------------------------------------
FAMILY: dict[str, dict[str, Any]] = {
    # images are pre-scaled to <= 2048 px (cache), so capping the processor at 2048x2048 pixels changes nothing for
    # our inputs but stops vLLM from reserving encoder memory for 16-megapixel images (more KV cache)
    "qwen3vl": {"family": "Qwen3-VL", "org": "Alibaba", "license": "apache-2.0",
                "vllm": {"limit_mm_per_prompt": {"image": 1, "video": 0},
                         "mm_processor_kwargs": {"size": {"longest_edge": 4194304, "shortest_edge": 65536}}},
                "autotune": {"mm_processor_kwargs": QWEN_MM}},
    "qwen2vl": {"vllm": {"limit_mm_per_prompt": {"image": 1, "video": 0}},
                "autotune": {"mm_processor_kwargs": QWEN2_MM}},
    "internvl": {"family": "InternVL3.5", "org": "Shanghai AI Lab", "license": "apache-2.0",
                 "vllm": {"trust_remote_code": True}, "autotune": {"mm_processor_kwargs": INTERNVL_MM},
                 "fallback_backend": "hf", "hf": {"adapter": "internvl_chat", "model_class": "AutoModel",
                                                   "max_num_tiles": 12}},
    "gemma4": {"family": "Gemma 4", "org": "Google", "license": "apache-2.0",
               "vllm": {"limit_mm_per_prompt": {"image": 1, "audio": 0, "video": 0},
                        "mm_processor_kwargs": {"max_soft_tokens": 1120}, "reasoning_parser": "gemma4"},
               "autotune": {"mm_processor_kwargs": [{"max_soft_tokens": 560}, {"max_soft_tokens": 280}]},
               "thinking": THINK_FLAG},
    "gemma4_small": {"vllm": {"max_num_seqs": 16}},   # E2B/E4B: few KV heads -> large KV cache
}

MISTRAL_HF_PATTERNS = ["*.json", "*.jinja", "model*.safetensors", "tokenizer*"]   # skip consolidated.safetensors

COMMON_VLM = {"role": "vlm", "backend": "vllm", "dtype": "bfloat16",
              "vllm": {"max_model_len": 16384, "max_num_seqs": 8, "gpu_memory_utilization": 0.90,
                       "limit_mm_per_prompt": {"image": 1}},
              "autotune": {"max_num_seqs": [16, 8, 4, 2, 1], "max_model_len": [16384, 12288, 8192]}}
SMALL = {"vllm": {"max_num_seqs": 16}}
COMMON_LLM = {"role": "llm", "backend": "vllm", "dtype": "bfloat16", "text_only": True,
              "vllm": {"max_model_len": 16384, "max_num_seqs": 16, "gpu_memory_utilization": 0.90,
                       "limit_mm_per_prompt": None},
              "autotune": {"max_num_seqs": [16, 8, 4, 2, 1], "max_model_len": [16384, 12288, 8192]}}
COMMON_OCR_VLM = {"role": "ocr", "backend": "vllm", "dtype": "bfloat16",
                  "vllm": {"max_model_len": 16384, "max_num_seqs": 16, "gpu_memory_utilization": 0.85,
                           "limit_mm_per_prompt": {"image": 1}},
                  "autotune": {"max_num_seqs": [16, 8, 4, 2], "max_model_len": [16384, 8192]}}
COMMON_OCR_CLASSIC = {"role": "ocr", "backend": "ocr", "hf_id": None, "download_gb": 0.6, "dtype": None}

# per-model presets ----------------------------------------------------------------------------------
# `base`: list of preset dicts merged in order; then the inline keys.
MODELS: dict[str, dict[str, Any]] = {
    # ---- G1 / G2 / G3: Qwen3-VL
    "qwen3vl_8b": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-8B-Instruct", "params_b": 8.8,
                   "priority": 10},
    "qwen3vl_2b": {"base": [COMMON_VLM, SMALL, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-2B-Instruct", "params_b": 2.1},
    "qwen3vl_4b": {"base": [COMMON_VLM, SMALL, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-4B-Instruct", "params_b": 4.4},
    "qwen3vl_8b_thinking": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-8B-Thinking",
                            "params_b": 8.8, "thinking": {"supported": True, "default_on": True, "off": {}, "on": {}},
                            "sampling": {"max_tokens": 12288},
                            "vllm": {"max_model_len": 20480, "max_num_seqs": 4},
                            "autotune": {"max_model_len": [20480, 16384, 12288]}},
    "qwen3vl_32b_awq": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "QuantTrio/Qwen3-VL-32B-Instruct-AWQ",
                        "params_b": 33, "quant": "awq", "dtype": "float16",
                        "vllm": {"max_num_seqs": 2, "max_model_len": 12288, "gpu_memory_utilization": 0.95},
                        "autotune": {"max_num_seqs": [2, 1], "max_model_len": [12288, 8192]},
                        "note": "community AWQ 4-bit (no official AWQ for 32B)"},
    "qwen3vl_8b_awq": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
                       "params_b": 8.8, "quant": "awq", "dtype": "float16", "groups": ["G6"], "tier": "full",
                       "note": "G6: community AWQ 4-bit of qwen3vl_8b"},
    "qwen3vl_8b_fp8": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-8B-Instruct-FP8",
                       "params_b": 8.8, "quant": "fp8-weight-only", "groups": ["G6"], "tier": "full",
                       "note": "G6: official FP8; Ampere has no FP8 cores, vLLM runs it weight-only (Marlin)"},
    "qwen3vl_8b_gguf_q4": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                           "enabled": False, "groups": ["G6"], "tier": "optional",
                           "allow_patterns": ["*Q4_K_M*.gguf", "*mmproj*F16*.gguf"],
                           "note": "disabled: multimodal GGUF needs a llama.cpp backend (Step 0, decision 7)"},
    "qwen25vl_7b": {"base": [COMMON_VLM, FAMILY["qwen2vl"]], "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "params_b": 8.3,
                    "family": "Qwen2.5-VL", "org": "Alibaba"},
    "qwen35_9b": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3.5-9B", "params_b": 9.7,
                  "family": "Qwen3.5", "groups": ["G1", "G3"], "tier": "optional", "thinking": THINK_DEFAULT_ON,
                  "vllm": {"max_num_seqs": 4, "max_model_len": 12288, "gpu_memory_utilization": 0.93},
                  "autotune": {"max_num_seqs": [4, 2, 1], "max_model_len": [12288, 8192]}},
    # ---- InternVL3.5
    "internvl35_8b": {"base": [COMMON_VLM, FAMILY["internvl"]], "hf_id": "OpenGVLab/InternVL3_5-8B", "params_b": 8.5,
                      "priority": 20},
    "internvl35_1b": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "OpenGVLab/InternVL3_5-1B", "params_b": 1.1},
    "internvl35_2b": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "OpenGVLab/InternVL3_5-2B", "params_b": 2.3},
    "internvl35_4b": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "OpenGVLab/InternVL3_5-4B", "params_b": 4.7},
    "internvl35_14b": {"base": [COMMON_VLM, FAMILY["internvl"]], "hf_id": "brandonbeiler/InternVL3_5-14B-FP8-Dynamic",
                       "params_b": 15, "quant": "fp8-weight-only",
                       "vllm": {"max_num_seqs": 4, "gpu_memory_utilization": 0.93},
                       "note": "community FP8 checkpoint (bf16 30 GB does not fit 24 GB)"},
    # ---- Gemma 4
    "gemma4_e4b": {"base": [COMMON_VLM, FAMILY["gemma4"], FAMILY["gemma4_small"]], "hf_id": "google/gemma-4-E4B-it", "params_b": 8,
                   "priority": 30},
    "gemma4_e2b": {"base": [COMMON_VLM, SMALL, FAMILY["gemma4"], FAMILY["gemma4_small"]], "hf_id": "google/gemma-4-E2B-it", "params_b": 5},
    "gemma4_12b": {"base": [COMMON_VLM, FAMILY["gemma4"]], "hf_id": "google/gemma-4-12B-it-qat-w4a16-ct",
                   "params_b": 12, "quant": "w4a16-qat", "note": "official QAT 4-bit; bf16 (23.95 GB) does not fit"},
    "gemma4_26b_a4b_4bit": {"base": [COMMON_VLM, FAMILY["gemma4"]], "hf_id": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                            "params_b": 26, "quant": "awq", "dtype": "bfloat16",
                            "vllm": {"max_num_seqs": 4, "gpu_memory_utilization": 0.93},
                            "note": "community AWQ 4-bit of the MoE (official bf16 is 52 GB)"},
    # ---- other families
    "glm46v_flash": {"base": [COMMON_VLM], "hf_id": "zai-org/GLM-4.6V-Flash", "params_b": 10.3, "family": "GLM-V",
                     "org": "Zhipu AI", "license": "mit", "priority": 40, "thinking": THINK_DEFAULT_ON,
                     # max_num_batched_tokens: vLLM's encoder cache is max(this, tokens of a square image at the pixel
                     # limit) = 2601 by default, and non-square images at the limit need up to 2646 tokens
                     "vllm": {"max_num_seqs": 8, "gpu_memory_utilization": 0.94, "max_num_batched_tokens": 4096,
                              "limit_mm_per_prompt": {"image": 1, "video": 0},
                              "mm_processor_kwargs": {"size": {"longest_edge": 4194304, "shortest_edge": 12544}}},
                     "autotune": {"max_num_seqs": [8, 4, 2, 1], "max_model_len": [16384, 12288, 8192],
                                  "mm_processor_kwargs": GLMV_MM}},
    "minicpmv45": {"base": [COMMON_VLM], "hf_id": "openbmb/MiniCPM-V-4_5", "params_b": 8.7, "family": "MiniCPM-V",
                   "org": "OpenBMB", "license": "MiniCPM model licence (apache-2.0 tag)", "priority": 50,
                   "thinking": THINK_FLAG, "vllm": {"trust_remote_code": True},
                   "fallback_backend": "hf", "hf": {"adapter": "minicpmv", "batch_size": 1, "batch_ladder": [1]}},
    "phi4_mm": {"base": [COMMON_VLM], "hf_id": "microsoft/Phi-4-multimodal-instruct", "params_b": 5.6,
                "family": "Phi-4", "org": "Microsoft", "license": "mit", "request_model": "vision",
                "vllm": {"trust_remote_code": True, "max_model_len": 12288,
                         "extra_args": ["--enable-lora", "--max-lora-rank", "320", "--max-loras", "1",
                                        "--lora-modules", "vision={model_dir}/vision-lora"]},
                "autotune": {"max_model_len": [12288, 8192]}},
    "pixtral_12b": {"base": [COMMON_VLM], "hf_id": "nm-testing/pixtral-12b-FP8-dynamic", "params_b": 12.4,
                    "family": "Pixtral", "org": "Mistral AI", "license": "apache-2.0", "quant": "fp8-weight-only",
                    "vllm": {"max_num_seqs": 4},
                    "note": "Neural Magic FP8 checkpoint of Pixtral-12B (bf16 25 GB does not fit)"},
    "molmo_7b": {"base": [COMMON_VLM], "hf_id": "allenai/Molmo-7B-D-0924", "params_b": 8, "family": "Molmo",
                 "org": "Allen AI", "license": "apache-2.0", "vllm": {"trust_remote_code": True},
                 "ignore_patterns": ["*.md"],
                 # chat template raises on a system role -> system prompt prepended to the user turn
                 "system_role": "merge"},
    "ovis25_9b": {"base": [COMMON_VLM], "hf_id": "AIDC-AI/Ovis2.5-9B", "params_b": 9.2, "family": "Ovis",
                  "org": "Alibaba AIDC", "license": "apache-2.0", "thinking": THINK_DEFAULT_ON,
                  "vllm": {"trust_remote_code": True, "max_num_seqs": 4}, "fallback_backend": "hf",
                  "hf": {"adapter": "generic", "model_class": "AutoModelForCausalLM", "batch_size": 1}},
    "aya_vision_8b": {"base": [COMMON_VLM], "hf_id": "unsloth/aya-vision-8b", "params_b": 8.6, "family": "Aya Vision",
                      "org": "Cohere", "license": "cc-by-nc-4.0", "backend": "hf",
                      "hf": {"adapter": "generic", "model_class": "AutoModelForImageTextToText", "batch_size": 4},
                      "note": "vLLM 0.32 has no AyaVision implementation -> transformers; ungated mirror of CohereLabs/aya-vision-8b"},
    # ---- extended families (2026-11-14): scale ladders and further families for the full study ----------------
    # Qwen3.5: natively multimodal, thinking on by default -> the main run sends enable_thinking=false
    "qwen35_0p8b": {"base": [COMMON_VLM, SMALL, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3.5-0.8B", "params_b": 0.9,
                    "family": "Qwen3.5", "groups": ["G2"], "tier": "full", "thinking": THINK_DEFAULT_ON},
    "qwen35_2b": {"base": [COMMON_VLM, SMALL, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3.5-2B", "params_b": 2.3,
                  "family": "Qwen3.5", "groups": ["G2"], "tier": "full", "thinking": THINK_DEFAULT_ON},
    "qwen35_4b": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "Qwen/Qwen3.5-4B", "params_b": 4.7,
                  "family": "Qwen3.5", "groups": ["G2"], "tier": "full", "thinking": THINK_DEFAULT_ON,
                  "vllm": {"max_num_seqs": 8}},
    # Ministral 3 (Mistral AI, vision): HF-format weights only (the repos also hold a second "consolidated" copy)
    "ministral3_3b": {"base": [COMMON_VLM, SMALL], "hf_id": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
                      "params_b": 3.8, "family": "Ministral 3", "org": "Mistral AI", "license": "apache-2.0",
                      "groups": ["G2"], "tier": "full", "allow_patterns": MISTRAL_HF_PATTERNS},
    "ministral3_8b": {"base": [COMMON_VLM], "hf_id": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "params_b": 8.9,
                      "family": "Ministral 3", "org": "Mistral AI", "license": "apache-2.0", "groups": ["G1", "G2"],
                      "tier": "full", "allow_patterns": MISTRAL_HF_PATTERNS},
    "ministral3_14b_fp8": {"base": [COMMON_VLM], "hf_id": "mistralai/Ministral-3-14B-Instruct-2512",
                           "params_b": 14, "family": "Ministral 3", "org": "Mistral AI", "license": "apache-2.0",
                           "quant": "fp8-weight-only", "groups": ["G2"], "tier": "optional",
                           "allow_patterns": MISTRAL_HF_PATTERNS,
                           "vllm": {"max_num_seqs": 4, "gpu_memory_utilization": 0.93},
                           "note": "official FP8 checkpoint (bf16 28 GB does not fit); Ampere runs it weight-only"},
    # large models that just fit 24 GB with 4-bit weights: few concurrent requests, long runs -> optional
    "qwen3vl_30b_a3b_awq": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
                            "params_b": 31, "quant": "awq", "dtype": "float16", "groups": ["G2"], "tier": "optional",
                            "vllm": {"max_num_seqs": 4, "max_model_len": 12288, "gpu_memory_utilization": 0.95},
                            "autotune": {"max_num_seqs": [4, 2, 1], "max_model_len": [12288, 8192]},
                            "note": "MoE (3B active), community AWQ 4-bit (bf16 61 GB)"},
    "qwen35_27b_awq": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit", "params_b": 27,
                       "family": "Qwen3.5", "quant": "awq", "dtype": "float16", "groups": ["G2"], "tier": "optional",
                       "thinking": THINK_DEFAULT_ON,
                       "vllm": {"max_num_seqs": 2, "max_model_len": 10240, "gpu_memory_utilization": 0.95},
                       "autotune": {"max_num_seqs": [2, 1], "max_model_len": [10240, 8192]},
                       "note": "community AWQ 4-bit; 20 GB of weights leave little KV cache"},
    "gemma4_31b_awq": {"base": [COMMON_VLM, FAMILY["gemma4"]], "hf_id": "QuantTrio/gemma-4-31B-it-AWQ", "params_b": 31,
                       "quant": "awq", "dtype": "float16", "groups": ["G2"], "tier": "optional",
                       "vllm": {"max_num_seqs": 2, "max_model_len": 10240, "gpu_memory_utilization": 0.95},
                       "autotune": {"max_num_seqs": [2, 1], "max_model_len": [10240, 8192]},
                       "note": "community AWQ 4-bit of the dense 31B (official QAT checkpoint is 23 GB)"},
    # further families
    "nemotron_vl_12b_fp8": {"base": [COMMON_VLM], "hf_id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
                            "params_b": 12.6, "family": "Nemotron-VL", "org": "NVIDIA", "license": "NVIDIA Open Model",
                            "quant": "modelopt-fp8", "groups": ["G1"], "tier": "optional",
                            "vllm": {"trust_remote_code": True, "max_num_seqs": 4, "gpu_memory_utilization": 0.93,
                                     "extra_args": ["--quantization", "modelopt"]},
                            "note": "hybrid Mamba-Transformer; official FP8 (bf16 24 GB does not fit)"},
    "smolvlm2_2b": {"base": [COMMON_VLM, SMALL], "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "params_b": 2.2,
                    "family": "SmolVLM2", "org": "Hugging Face", "license": "apache-2.0", "groups": ["G1"],
                    "tier": "optional", "vllm": {"max_model_len": 8192}, "autotune": {"max_model_len": [8192]}},
    "granite_vision_4b": {"base": [COMMON_VLM, SMALL], "hf_id": "ibm-granite/granite-vision-4.1-4b", "params_b": 4.2,
                          "family": "Granite Vision", "org": "IBM", "license": "apache-2.0", "groups": ["G1"],
                          "tier": "optional"},
    "mimo_vl_7b": {"base": [COMMON_VLM, FAMILY["qwen2vl"]], "hf_id": "XiaomiMiMo/MiMo-VL-7B-RL-2508", "params_b": 8.3,
                   "family": "MiMo-VL", "org": "Xiaomi", "license": "mit", "groups": ["G1"], "tier": "optional"},
    # ---- G4 Vietnamese
    "vintern_1b": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "5CD-AI/Vintern-1B-v3_5", "params_b": 0.9,
                   "family": "Vintern", "org": "5CD-AI", "license": "mit", "priority": 60,
                   "hf": {"adapter": "internvl_chat", "model_class": "AutoModel", "max_num_tiles": 6}},
    "vintern_3b_r": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "5CD-AI/Vintern-3B-R-beta",
                     "params_b": 3.7, "family": "Vintern", "org": "5CD-AI", "license": "mit", "groups": ["G4"],
                     "tier": "full", "hf": {"adapter": "internvl_chat", "model_class": "AutoModel", "max_num_tiles": 6}},
    "vintern_3b": {"base": [COMMON_VLM, SMALL, FAMILY["internvl"]], "hf_id": "5CD-AI/Vintern-3B-beta", "params_b": 3.7,
                   "family": "Vintern", "org": "5CD-AI", "license": "unknown (no tag on card)", "groups": ["G4"],
                   "tier": "optional", "hf": {"adapter": "internvl_chat", "model_class": "AutoModel", "max_num_tiles": 6}},
    "erax_vl_7b": {"base": [COMMON_VLM, FAMILY["qwen2vl"]], "hf_id": "erax-ai/EraX-VL-7B-V1.5", "params_b": 8.3,
                   "family": "EraX-VL (Qwen2-VL)", "org": "EraX", "license": "apache-2.0", "groups": ["G4"],
                   "tier": "pilot", "priority": 65},
    "erax_vl_2b": {"base": [COMMON_VLM, SMALL, FAMILY["qwen2vl"]], "hf_id": "erax-ai/EraX-VL-2B-V1.5", "params_b": 2.2,
                   "family": "EraX-VL (Qwen2-VL)", "org": "EraX", "license": "apache-2.0", "groups": ["G4"], "tier": "full"},
    "erax_vl_7b_v2": {"base": [COMMON_VLM, FAMILY["qwen2vl"]], "hf_id": "erax-ai/EraX-VL-7B-V2.0-Preview",
                      "params_b": 8.3, "family": "EraX-VL (Qwen2-VL)", "org": "EraX", "license": "apache-2.0",
                      "groups": ["G4"], "tier": "optional"},
    "qwen3vl_8b_vi_ocrvqa": {"base": [COMMON_VLM, FAMILY["qwen3vl"]], "hf_id": "phucdu/Qwen3-VL-8B-Vietnamese-OCR-VQA",
                             "params_b": 8.8, "family": "Qwen3-VL (community VI fine-tune)", "org": "community",
                             "license": "unknown (no card)", "groups": ["G4"], "tier": "optional"},
    # ---- G5 OCR (classic engines run in the OCR venv)
    "ocr_ppocr": {"base": [COMMON_OCR_CLASSIC], "family": "PaddleOCR PP-OCRv5", "org": "Baidu", "license": "apache-2.0",
                  "ocr": {"engine": "paddleocr", "lang": "vi"}, "tier": "pilot", "priority": 1},
    "ocr_easyocr": {"base": [COMMON_OCR_CLASSIC], "family": "EasyOCR", "org": "JaidedAI", "license": "apache-2.0",
                    "ocr": {"engine": "easyocr", "options": {"langs": ["vi", "en"], "gpu": False}},
                    "note": "runs on CPU (torch CPU in the OCR venv); weights from GitHub releases"},
    "ocr_vietocr": {"base": [COMMON_OCR_CLASSIC], "family": "VietOCR (+PP-OCRv5 det)", "org": "pbcquoc",
                    "license": "apache-2.0", "groups": ["G4", "G5"], "tier": "full",
                    "ocr": {"engine": "ppocr_vietocr", "options": {"vietocr_config": "vgg_transformer",
                                                                   "rec_device": "cpu"}},
                    "note": "VietOCR recognises single lines; PaddleOCR detects them. Weights from vocr.vn"},
    "ocr_paddleocr_vl": {"base": [COMMON_OCR_VLM], "hf_id": "PaddlePaddle/PaddleOCR-VL", "params_b": 0.9,
                         "family": "PaddleOCR-VL", "org": "Baidu", "license": "apache-2.0",
                         "vllm": {"trust_remote_code": True, "max_num_batched_tokens": 16384,
                                  "extra_args": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]},
                         "ocr": {"engine": "vlm_prompt", "prompt": "OCR:"}},
    "ocr_glm_ocr": {"base": [COMMON_OCR_VLM], "hf_id": "zai-org/GLM-OCR", "params_b": 0.9, "family": "GLM-OCR",
                    "org": "Zhipu AI", "license": "mit", "thinking": {"supported": False, "default_on": False,
                                                                      "off": {"chat_template_kwargs": {"enable_thinking": False}}},
                    "ocr": {"engine": "vlm_prompt", "prompt": "Text Recognition:"}},
    "ocr_deepseek_ocr2": {"base": [COMMON_OCR_VLM], "hf_id": "deepseek-ai/DeepSeek-OCR-2", "params_b": 3,
                          "family": "DeepSeek-OCR", "org": "DeepSeek", "license": "apache-2.0",
                          "vllm": {"trust_remote_code": True, "max_model_len": 8192,
                                   "extra_args": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]},
                          "autotune": {"max_model_len": [8192]}, "sampling": {"skip_special_tokens": False},
                          "ocr": {"engine": "vlm_prompt", "prompt": "Free OCR.", "max_tokens": 6144},
                          "system_role": "drop"},
    "ocr_deepseek_ocr": {"base": [COMMON_OCR_VLM], "hf_id": "deepseek-ai/DeepSeek-OCR", "params_b": 3,
                         "family": "DeepSeek-OCR", "org": "DeepSeek", "license": "mit", "tier": "optional",
                         "groups": ["G5"],
                         "vllm": {"trust_remote_code": True, "max_model_len": 8192,
                                  "extra_args": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]},
                         "autotune": {"max_model_len": [8192]},
                         "ocr": {"engine": "vlm_prompt", "prompt": "Free OCR.", "max_tokens": 6144},
                         "system_role": "drop"},
    "ocr_dots": {"base": [COMMON_OCR_VLM], "hf_id": "rednote-hilab/dots.ocr", "params_b": 3, "family": "dots.ocr",
                 "org": "rednote", "license": "mit", "vllm": {"trust_remote_code": True},
                 "ocr": {"engine": "vlm_prompt", "prompt": "Extract the text content from this image."}},
    "ocr_mineru25": {"base": [COMMON_OCR_VLM], "hf_id": "opendatalab/MinerU2.5-2509-1.2B", "params_b": 1.2,
                     "family": "MinerU2.5", "org": "OpenDataLab", "license": "agpl-3.0",
                     "vllm": {"max_model_len": 16384},
                     "ocr": {"engine": "vlm_prompt", "prompt": "\nText Recognition:"}},
    "ocr_glm_ocr_vi": {"base": [COMMON_OCR_VLM], "hf_id": "KuongB/glm-ocr-vietnamese", "params_b": 0.9,
                       "family": "GLM-OCR (community VI fine-tune)", "org": "community", "license": "unknown",
                       "groups": ["G4", "G5"], "tier": "optional",
                       "thinking": {"supported": False, "off": {"chat_template_kwargs": {"enable_thinking": False}}},
                       "ocr": {"engine": "vlm_prompt", "prompt": "Text Recognition:"}},
    # ---- G5 text LLMs
    "llm_qwen3_8b": {"base": [COMMON_LLM], "hf_id": "Qwen/Qwen3-8B", "params_b": 8.2, "family": "Qwen3",
                     "org": "Alibaba", "license": "apache-2.0", "thinking": THINK_DEFAULT_ON},
    "llm_gemma3_12b_text": {"base": [COMMON_LLM], "hf_id": "RedHatAI/gemma-3-12b-it-FP8-dynamic", "params_b": 12,
                            "family": "Gemma 3", "org": "Google", "license": "gemma", "quant": "fp8-weight-only",
                            "vllm": {"max_num_seqs": 8, "extra_args": ["--limit-mm-per-prompt", "{\"image\": 0}"]},
                            "note": "Red Hat FP8 checkpoint (ungated); bf16 24.4 GB does not fit"},
    "llm_vylinh_3b": {"base": [COMMON_LLM], "hf_id": "arcee-ai/Arcee-VyLinh", "params_b": 3.1, "family": "VyLinh (Qwen2.5)",
                      "org": "Arcee AI", "license": "apache-2.0", "groups": ["G4", "G5"], "tier": "full",
                      "vllm": {"max_model_len": 16384}},
    "llm_sailor2_8b": {"base": [COMMON_LLM], "hf_id": "sail/Sailor2-8B-Chat", "params_b": 8.9, "family": "Sailor2",
                       "org": "Sea AI Lab", "license": "apache-2.0", "groups": ["G4", "G5"], "tier": "full"},
    "llm_seallm_v3_7b": {"base": [COMMON_LLM], "hf_id": "SeaLLMs/SeaLLMs-v3-7B-Chat", "params_b": 7.6,
                         "family": "SeaLLMs", "org": "DAMO", "license": "seallms licence", "groups": ["G4", "G5"],
                         "tier": "optional"},
    "llm_phogpt_4b": {"base": [COMMON_LLM], "hf_id": "vinai/PhoGPT-4B-Chat", "params_b": 3.7, "family": "PhoGPT",
                      "org": "VinAI", "license": "bsd-3-clause", "groups": ["G4", "G5"], "tier": "optional",
                      "backend": "hf", "vllm": {"trust_remote_code": True, "max_model_len": 8192},
                      "hf": {"adapter": "generic", "model_class": "AutoModelForCausalLM", "batch_size": 4},
                      # chat template silently drops system messages -> prepend to the user turn
                      "system_role": "merge",
                      "note": "MPT architecture has no vLLM implementation -> transformers"},
    "llm_vistral_7b": {"base": [COMMON_LLM], "hf_id": "Viet-Mistral/Vistral-7B-Chat", "params_b": 7.3,
                       "family": "Vistral", "org": "Viet-Mistral", "license": "afl-3.0", "groups": ["G4", "G5"],
                       "tier": "optional", "gated": "manual", "enabled": False,
                       "note": "dropped 2026-11-02: gated repo, no HF token (user decision)"},
}


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(key: str, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    preset = dict(MODELS.get(key, {}))
    bases = preset.pop("base", [COMMON_VLM])
    spec: dict[str, Any] = {}
    for b in bases:
        spec = merge(spec, b)
    cand = dict(candidate or {})
    if cand:
        spec = merge(spec, {k: v for k, v in {
            "family": cand.get("family"), "params_b": cand.get("params_b"), "groups": cand.get("group"),
            "tier": cand.get("tier"), "license": cand.get("license"), "note": cand.get("note"),
            "text_only": cand.get("text_only"), "download_gb": cand.get("size_gb"),
        }.items() if v is not None})
        if cand.get("backend") and cand["backend"] in ("vllm", "hf", "ocr") and key not in MODELS:
            spec["backend"] = cand["backend"]
        if cand.get("hf_id") and " " not in str(cand["hf_id"]) and "pip " not in str(cand["hf_id"]) and key not in MODELS:
            spec["hf_id"] = cand["hf_id"]
    spec = merge(spec, preset)
    spec["key"] = key
    if spec.get("limit_mm_per_prompt") is None and spec.get("vllm", {}).get("limit_mm_per_prompt") is None:
        spec.setdefault("vllm", {})["limit_mm_per_prompt"] = None
    spec.setdefault("groups", [])
    spec.setdefault("tier", "full")
    return spec
