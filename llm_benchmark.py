import os
import time
import subprocess
import statistics
from tabulate import tabulate
from vllm import LLM, SamplingParams

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODELS = [
    "./models/Mistral-7B-Instruct-v0.3",
    "./models/gemma-7b-it",
    "./models/Qwen2-7B-Instruct",
    "./models/gpt-oss-20b"     
]

TEST_PROMPTS = {
    "coding": (
        "Write a Python function to calculate Fibonacci numbers. "
        "Provide both recursive and iterative implementations, "
        "then explain their time and space complexity in detail. "
        "Also, extend the solution to include memoization and explain how "
        "it improves performance compared to naive recursion."
    ),
    "creative": (
        "Write a short poem about the moon in 3 lines. "
        "Expand it with another 3 lines describing the moonlight reflecting on the ocean. "
        "Then, add 3 more lines using metaphors about dreams and time."
    ),
    "knowledge": (
        "What is 12 multiplied by 8? "
        "Show the step by step arithmetic. "
        "Then calculate (15*7) - (9*3) + (144/12). "
        "Explain each step clearly so that even a beginner can follow."
    ),
    "reasoning": (
        "If there are 5 apples and you eat 2, how many are left? "
        "Now imagine you buy 7 more apples and share them equally with 2 friends. "
        "Explain step by step how many apples each person gets."
    )
}

MAX_NEW_TOKENS = 256
OUTPUT_TXT = "benchmark_results.txt"

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_gpu_stats():
    """Collect GPU utilization and VRAM usage for all GPUs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    lines = result.stdout.strip().split("\n")
    utils, mems = [], []
    for line in lines:
        gpu_util, mem_used = line.split(", ")
        utils.append(float(gpu_util))
        mems.append(float(mem_used))
    return utils, mems

def accuracy_score(answer, expected):
    """Calculate simple accuracy as fuzzy match percent (0~100)."""
    if expected is None:
        return "N/A"
    answer, expected = answer.lower(), expected.lower()
    if expected in answer:
        return 100.0
    match_count = sum(1 for w in expected.split() if w in answer)
    return round(100.0 * match_count / len(expected.split()), 1)

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def run_test(model_name, test_type, prompt):
    print(f"\n=== Running {model_name} | {test_type} ===")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
        max_model_len=1024,
        max_num_seqs=32,
        disable_log_stats=True
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # GPU monitor thread
    utils_samples, mems_samples = [], []
    stop_flag = False

    def monitor():
        while not stop_flag:
            u, m = get_gpu_stats()
            utils_samples.append(u)
            mems_samples.append(m)
            time.sleep(0.2)

    import threading
    t = threading.Thread(target=monitor)
    t.start()

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    stop_flag = True
    t.join()

    total_time = end_time - start_time
    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)

    # Approximate TTFT: total_time ÷ number of tokens
    ttft = round(total_time / max(tokens, 1), 2)
    tps = tokens / total_time if total_time > 0 else 0

    # GPU statistics
    gpu_avgs, gpu_maxs, vram_maxs = [], [], []
    if utils_samples:
        for g in range(len(utils_samples[0])):
            vals = [sample[g] for sample in utils_samples]
            memv = [sample[g] for sample in mems_samples]
            gpu_avgs.append(round(statistics.mean(vals), 1))
            gpu_maxs.append(max(vals))
            vram_maxs.append(max(memv))
    else:
        gpu_avgs, gpu_maxs, vram_maxs = ["N/A"], ["N/A"], ["N/A"]

    expected = None
    if test_type == "knowledge":
        expected = "117"
    elif test_type == "reasoning":
        expected = "5"
    acc = accuracy_score(text, expected)

    return {
        "Model": os.path.basename(model_name),
        "Test Type": test_type,
        "Total Time": round(total_time, 2),
        "TTFT": ttft,
        "TPS": round(tps, 2),
        "GPU Count": len(gpu_avgs),
        "GPU Avg": max(gpu_avgs) if gpu_avgs != ["N/A"] else "N/A",
        "GPU Max": max(gpu_maxs) if gpu_maxs != ["N/A"] else "N/A",
        "VRAM Max": max(vram_maxs) if vram_maxs != ["N/A"] else "N/A",
        "Length": len(text.split()),
        "Tokens": tokens,
        "Accuracy": acc,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    results = []
    for model in MODELS:
        for test_type, prompt in TEST_PROMPTS.items():
            res = run_test(model, test_type, prompt)
            results.append(res)

    print("\n=== Summary Results ===")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    with open(OUTPUT_TXT, "w") as f:
        f.write("=== Summary Results ===\n")
        f.write(tabulate(results, headers="keys", tablefmt="grid"))

    print(f"\nResults saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
