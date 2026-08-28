import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUTPUT_DIR = "models"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 500_000_000:
        print(f"Model already exists at {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE):,} bytes).")
        return

    print(f"Downloading Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~986 MB)...")
    print(f"From: {URL}")
    print(f"To:   {OUTPUT_FILE}")

    start_time = time.time()
    last_print = [start_time]

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        now = time.time()
        if now - last_print[0] >= 1.0 or (total_size > 0 and downloaded >= total_size):
            last_print[0] = now
            elapsed = now - start_time
            speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
            pct = (downloaded / total_size * 100) if total_size > 0 else 0
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  [{pct:5.1f}%] {mb_down:6.1f} MB / {mb_total:6.1f} MB ({speed:5.1f} MB/s)", end="", flush=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(URL, headers=headers)
    
    with urllib.request.urlopen(req) as resp, open(OUTPUT_FILE, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        block_size = 1024 * 1024
        block_num = 0
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            block_num += 1
            reporthook(block_num, block_size, total_size)

    total_time = time.time() - start_time
    print(f"\nDownload completed in {total_time:.2f}s ({os.path.getsize(OUTPUT_FILE):,} bytes).")

if __name__ == "__main__":
    main()
