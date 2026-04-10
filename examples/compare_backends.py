"""Set up and compare different LLM serving backends.

Demonstrates OpenAIBackend, VLLMBackend, TGIBackend, and detect_backend.
Each backend sends a single test request to its server.
"""

from infermark import OpenAIBackend, VLLMBackend, TGIBackend, detect_backend


def main() -> None:
    # -- OpenAI-compatible backend (works with vLLM, Ollama, etc.) --------
    print("=== OpenAI Backend ===")
    openai = OpenAIBackend(
        url="http://localhost:8000",
        model="meta-llama/Llama-3-8B-Instruct",
        api_key="",  # no key needed for local servers
    )
    print(f"  Backend: {openai.backend_name}")
    result = openai.send_request("Say hello.", {"max_tokens": 32, "temperature": 0.7})
    print(f"  Success: {result.success}  Latency: {result.latency:.3f}s")
    if result.error:
        print(f"  Error: {result.error}")
    print()

    # -- vLLM native backend ----------------------------------------------
    print("=== vLLM Backend ===")
    vllm = VLLMBackend(url="http://localhost:8000", model="llama-3-8b")
    print(f"  Backend: {vllm.backend_name}")
    result = vllm.send_request("What is 2+2?", {"max_tokens": 16})
    print(f"  Success: {result.success}  Latency: {result.latency:.3f}s")
    if result.error:
        print(f"  Error: {result.error}")
    print()

    # -- TGI backend ------------------------------------------------------
    print("=== TGI Backend ===")
    tgi = TGIBackend(url="http://localhost:8080", model="llama-3-8b")
    print(f"  Backend: {tgi.backend_name}")
    result = tgi.send_request("Explain gravity.", {"max_tokens": 64})
    print(f"  Success: {result.success}  Latency: {result.latency:.3f}s")
    if result.error:
        print(f"  Error: {result.error}")
    print()

    # -- Auto-detect backend from URL -------------------------------------
    print("=== Auto-detect Backend ===")
    backend = detect_backend("http://localhost:8000", model="auto")
    print(f"  Detected: {backend.backend_name}")


if __name__ == "__main__":
    main()
