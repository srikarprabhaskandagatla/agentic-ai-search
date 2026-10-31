import uvicorn

from backend.main import app

try:
    import spaces

    @spaces.GPU
    def zero_gpu_check():
        return None

    zero = getattr(spaces, "zero", None)
    if zero is not None and hasattr(zero, "startup"):
        zero.startup()
except ImportError:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
