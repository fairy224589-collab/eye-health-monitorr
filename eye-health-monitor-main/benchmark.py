import time
import numpy as np
 
try:
    import onnxruntime as rt
except ImportError:
    raise SystemExit("onnxruntime not found. Run: pip install onnxruntime")
 
# ── CONFIG — edit these two paths ─────────────────────────────────────────────
FP32_MODEL = "models/model.onnx"   # your original FP32 model
INT8_MODEL = "models/model_int8.onnx"   # your quantized INT8 model
INPUT_SHAPE = (1, 3, 112, 112)         # must match what your model expects
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50
# ──────────────────────────────────────────────────────────────────────────────
 
 
def benchmark(model_path: str, input_shape: tuple,
              warmup: int, runs: int) -> dict:
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1   # single-thread = fair CPU comparison
    sess_options.inter_op_num_threads = 1
 
    sess = rt.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
 
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)
 
    # warmup — excluded from timing
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})
 
    # timed runs
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times_ms.append((time.perf_counter() - t0) * 1000)
 
    return {
        "avg_ms":    round(sum(times_ms) / len(times_ms), 2),
        "min_ms":    round(min(times_ms), 2),
        "max_ms":    round(max(times_ms), 2),
        "p95_ms":    round(sorted(times_ms)[int(0.95 * len(times_ms))], 2),
    }
 
 
def main():
    print(f"\nInput shape : {INPUT_SHAPE}")
    print(f"Warmup runs : {WARMUP_RUNS}")
    print(f"Bench runs  : {BENCHMARK_RUNS}")
    print(f"Provider    : CPUExecutionProvider (single-thread)\n")
    print("─" * 52)
 
    results = {}
    for label, path in [("FP32", FP32_MODEL), ("INT8", INT8_MODEL)]:
        print(f"Benchmarking {label}  →  {path}")
        try:
            results[label] = benchmark(path, INPUT_SHAPE, WARMUP_RUNS, BENCHMARK_RUNS)
            r = results[label]
            print(f"  avg : {r['avg_ms']} ms")
            print(f"  min : {r['min_ms']} ms")
            print(f"  max : {r['max_ms']} ms")
            print(f"  p95 : {r['p95_ms']} ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[label] = None
        print()
 
    print("─" * 52)
 
    if results.get("FP32") and results.get("INT8"):
        fp32_avg = results["FP32"]["avg_ms"]
        int8_avg = results["INT8"]["avg_ms"]
        delta_ms = round(fp32_avg - int8_avg, 2)
        speedup  = round(fp32_avg / int8_avg, 2)
 
        print(f"\n📊 SUMMARY")
        print(f"  FP32 avg latency : {fp32_avg} ms")
        print(f"  INT8 avg latency : {int8_avg} ms")
        print(f"  Latency reduction: {delta_ms} ms faster")
        print(f"  Speedup          : {speedup}×")
        print(f"\n✅ Resume bullet numbers:")
        print(f"  → 'reduced inference latency by {delta_ms}ms ({speedup}× speedup)'")
        print(f"  → 'INT8 inference averaged {int8_avg}ms on CPU vs {fp32_avg}ms FP32 baseline'\n")
    else:
        print("\nCould not compute comparison — check model paths above.\n")
 
 
if __name__ == "__main__":
    main()
 
