# LLM GPU Benchmark Tool

This repository provides a lightweight benchmarking framework for evaluating large language models (LLMs) on **multi-GPU environments**.  
It is designed to measure **performance, GPU utilization, and basic accuracy** across different types of tasks.  

The tool has been tested on **Akamai RTX 4000 Ada (4 GPUs)** but can be adapted to other GPU setups.

---

## ✨ Features

- **Multi-Model Support**: Run benchmarks across multiple LLMs (local models only).  
- **Test Categories**: Coding, Creative writing, Knowledge Q&A, and Reasoning.  
- **Metrics Collected**:
  - Total inference time  
  - Time to First Token (TTFT, approximated)  
  - Tokens per second (TPS)  
  - GPU utilization (average & max)  
  - VRAM usage (max)  
  - Token count and output length  
  - Accuracy (fuzzy-matching % for selected tasks)  
- **Automatic Results Export**: Results saved in both console output and `benchmark_results.txt`.

---

## 📂 Project Structure

```
.
├── original.py             # Main benchmark script
├── benchmark_results.txt   # Output results (generated after run)
├── models/                 # Local models directory
│   ├── Mistral-7B-Instruct-v0.3
│   ├── gemma-7b-it
│   ├── Qwen2-7B-Instruct
│   └── gpt-oss-20b
```

---

## 🚀 Usage

### 1. Prepare Local Models
Download models into the `./models/` directory. Example:
- `./models/Mistral-7B-Instruct-v0.3`
- `./models/gemma-7b-it`
- `./models/Qwen2-7B-Instruct`
- `./models/gpt-oss-20b`

Each model folder should contain a `config.json`, `tokenizer.json`, and `safetensors` files.

### 2. Install Dependencies
```bash
pip install vllm tabulate
```

Ensure `nvidia-smi` is available for GPU monitoring.

### 3. Run the Benchmark
```bash
python original.py
```

### 4. Check Results
- Console output will show a formatted summary table.  
- Full results are written to `benchmark_results.txt`.

---

## 📊 Example Output

```
=== Summary Results ===
+--------------------------+-------------+--------------+--------+-------+-------------+-----------+-----------+------------+----------+----------+------------+
| Model                    | Test Type   |   Total Time |   TTFT |   TPS |   GPU Count |   GPU Avg |   GPU Max |   VRAM Max |   Length |   Tokens | Accuracy   |
+==========================+=============+==============+========+=======+=============+===========+===========+============+==========+==========+============+
| Mistral-7B-Instruct-v0.3 | coding      |         3.34 |   0.01 | 76.61 |           4 |      94   |        94 |      16154 |       84 |      256 | N/A        |
| ...                      | ...         |          ... |   ...  |  ...  |         ... |      ...  |       ... |       ...  |      ... |      ... | ...        |
```

---

## 📐 Accuracy Measurement

- Accuracy is calculated only for **knowledge** and **reasoning** tests.  
- Method: fuzzy word-match percentage between model output and the expected answer.  
- Other categories (`coding`, `creative`) are reported as **N/A**.

---

## ⚙️ Configuration

You can adjust these parameters in `original.py`:
- `MAX_NEW_TOKENS`: maximum tokens to generate (default: 256)  
- `tensor_parallel_size`: number of GPUs to use (default: 4)  
- `max_model_len`: maximum sequence length (default: 1024)  
- `TEST_PROMPTS`: customizable benchmark prompts per category  

---

## 📌 Notes

- This tool runs **locally**, no remote inference is used.  
- Results can vary significantly depending on:
  - Prompt complexity  
  - Model type and size  
  - GPU availability and parallelization  

---

## 📧 Contact

If you have questions, feel free to reach out or open an issue.  
