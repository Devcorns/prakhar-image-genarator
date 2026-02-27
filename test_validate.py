"""
Quick validation tests — verifies all project modules without downloading models.
Run with: python test_validate.py
"""

import warnings
warnings.filterwarnings("ignore")

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1


# ── Test 1: All imports ─────────────────────────────────────────────
def t1():
    from src.config import AppConfig, Precision, Scheduler
    from src.generator import TextToImageGenerator
    from src.optimizations import get_device, get_torch_dtype, estimate_vram_usage
    from src.schedulers import build_scheduler, SCHEDULER_MAP
    from src.prepare_dataset import resize_and_crop

test("All imports", t1)


# ── Test 2: Config defaults ─────────────────────────────────────────
def t2():
    from src.config import AppConfig, Precision, Scheduler
    config = AppConfig()
    assert config.model.model_id == "runwayml/stable-diffusion-v1-5"
    assert config.model.precision == Precision.FP16
    assert config.generation.width == 512
    assert config.generation.scheduler == Scheduler.DPM_PP_2M_KARRAS
    assert config.optimization.enable_attention_slicing is True
    assert config.lora.rank == 4

test("Config defaults", t2)


# ── Test 3: Device detection ────────────────────────────────────────
def t3():
    import torch
    from src.optimizations import get_device, get_torch_dtype
    device = get_device()
    dtype = get_torch_dtype("fp16", device)
    if device.type == "cpu":
        assert dtype == torch.float32, "CPU should force fp32"

test("Device detection + dtype", t3)


# ── Test 4: VRAM estimation ─────────────────────────────────────────
def t4():
    from src.optimizations import estimate_vram_usage
    est = estimate_vram_usage(512, 512, "fp16", cpu_offload=True)
    assert "estimated_peak_gb" in est
    assert est["estimated_peak_gb"] > 0

test("VRAM estimation", t4)


# ── Test 5: All schedulers registered ───────────────────────────────
def t5():
    from src.schedulers import SCHEDULER_MAP
    expected = ["euler_a", "euler", "dpm++_2m", "dpm++_2m_karras", "ddim", "lms", "pndm"]
    for name in expected:
        assert name in SCHEDULER_MAP, f"Missing: {name}"
    assert len(SCHEDULER_MAP) == 7

test("All 7 schedulers registered", t5)


# ── Test 6: Generator initialization ────────────────────────────────
def t6():
    from src.config import AppConfig
    from src.generator import TextToImageGenerator
    config = AppConfig()
    gen = TextToImageGenerator(config)
    assert gen.pipe is None
    assert gen.device is not None

test("Generator init (no model load)", t6)


# ── Test 7: CLI argument parsing ────────────────────────────────────
def t7():
    import sys
    original_argv = sys.argv
    sys.argv = ["main.py", "a beautiful sunset", "--steps", "20", "--seed", "42", "--estimate-vram"]
    from main import parse_args
    args = parse_args()
    assert args.prompt == "a beautiful sunset"
    assert args.steps == 20
    assert args.seed == 42
    assert args.estimate_vram is True
    sys.argv = original_argv

test("CLI argument parsing", t7)


# ── Test 8: Image resize/crop utility ───────────────────────────────
def t8():
    from PIL import Image
    from src.prepare_dataset import resize_and_crop
    img = Image.new("RGB", (1024, 768), color="red")
    cropped = resize_and_crop(img, 512)
    assert cropped.size == (512, 512)

test("Image resize + center crop", t8)


# ── Test 9: Generate raises without model ────────────────────────────
def t9():
    from src.config import AppConfig
    from src.generator import TextToImageGenerator
    gen = TextToImageGenerator(AppConfig())
    try:
        gen.generate("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "load_model" in str(e).lower() or "not loaded" in str(e).lower()

test("Generate raises without loaded model", t9)


# ── Test 10: Flush memory runs safely ───────────────────────────────
def t10():
    from src.optimizations import flush_memory
    flush_memory()  # Should not crash even without CUDA

test("flush_memory() on CPU", t10)


# ── Summary ─────────────────────────────────────────────────────────
print()
print("=" * 50)
print(f"  {passed} passed, {failed} failed  ({passed + failed} total)")
print("=" * 50)

if failed > 0:
    exit(1)
