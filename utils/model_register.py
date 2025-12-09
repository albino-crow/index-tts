from indextts.infer_v2 import IndexTTS2
from settings import USE_FLOATING_POINT_16

tts = IndexTTS2(
    model_dir="./checkpoints",
    cfg_path="./checkpoints/config.yaml",
    use_fp16=USE_FLOATING_POINT_16,
    use_deepspeed=False,
    use_cuda_kernel=False,
)


def get_model():
    try:
        yield tts
    except Exception as e:
        raise e
    finally:
        pass
