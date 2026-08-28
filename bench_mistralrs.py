import time
from mistralrs import Runner, Which, ChatCompletionRequest

def bench():
    print("Initializing mistral.rs (Rust Engine) for Llama-3.2-1B-Instruct...")
    r = Runner(
        Which.GGUF(
            quantized_model_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            quantized_filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            tok_model_id="unsloth/Llama-3.2-1B-Instruct"
        )
    )
    
    prompt = "Give me a numbered list of 5 historical dates."
    req = ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=41,
        temperature=0.0
    )
    
    print("Generating with mistral.rs...")
    t0 = time.perf_counter()
    resp = r.send_chat_completion_request(req)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    usage = resp.usage
    gen_tokens = usage.completion_tokens
    prefill_tokens = usage.prompt_tokens
    tok_per_sec = gen_tokens / total_time
    ms_per_tok = (total_time * 1000.0) / gen_tokens
    
    print(f"\n================================================================================")
    print(f"mistral.rs (Pure Rust Engine) Result:")
    print(f"Throughput: {tok_per_sec:.1f} tok/s | Latency: {ms_per_tok:.2f} ms/tok | Tokens: {gen_tokens}")
    print(f"Usage: Prompt tokens = {prefill_tokens}, Completion tokens = {gen_tokens}")
    print(f"Content: {resp.choices[0].message.content}")
    print(f"================================================================================\n")

if __name__ == "__main__":
    bench()
