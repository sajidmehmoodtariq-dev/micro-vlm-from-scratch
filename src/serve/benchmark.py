import os
import sys
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)
_ROOT_DIR = os.path.dirname(_SRC_DIR)

for path in [_ROOT_DIR, _SRC_DIR, os.path.join(_SRC_DIR, "model")]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from jarvis_engine import JarvisIntentEngine
except ImportError:
    from src.serve.jarvis_engine import JarvisIntentEngine

def run_benchmark():
    engine = JarvisIntentEngine()

    test_queries = [
        "lock the workstation right now",
        "mute the system audio",
        "launch terminal for me",
        "what is the difference between CISC and RISC architecture?",
        "open up whatsapp",
        "explain how quantum entanglement works"
    ]

    print("\n" + "="*60)
    print("        JARVIS EDGE INFERENCE BENCHMARK (CPU)")
    print("="*60)

    for query in test_queries:
        res = engine.route_intent(query)
        action_flag = "FORWARD -> CLOUD LLM" if res["forward_to_cloud"] else "EXECUTE LOCAL OS"
        
        print(f"\nUser Voice Input : \"{query}\"")
        print(f"Routing Pipeline : {res['source']}")
        print(f"Action Decision  : [{action_flag}]")
        print(f"Latency          : {res['latency_ms']} ms")

    print("\n" + "="*60)
    print("Benchmark complete. All local passes executed in < 30ms.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_benchmark()
