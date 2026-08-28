import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def bench():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_id} in PyTorch (FP16, SDPA, CUDA)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        attn_implementation="sdpa"
    ).to("cuda")
    
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        
    n_tokens = 41
    print(f"Generating {n_tokens} tokens with PyTorch SDPA...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tok_per_sec = gen_tokens / total_time
    ms_per_tok = (total_time * 1000.0) / gen_tokens
    
    print(f"\n================================================================================")
    print(f"PyTorch / Transformers (FP16 + SDPA Native) Result:")
    print(f"Throughput: {tok_per_sec:.1f} tok/s | Latency: {ms_per_tok:.2f} ms/tok | Tokens: {gen_tokens}")
    print(f"================================================================================\n")

if __name__ == "__main__":
    bench()
