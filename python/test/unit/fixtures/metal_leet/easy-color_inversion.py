import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image_ptr, width, height, BLOCK_SIZE: tl.constexpr):
    image_ptr = image_ptr.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)
    mask = (offsets < width * height * 4) & (offsets % 4 != 3)
    image = tl.load(image_ptr + offsets, mask=mask)
    image = 255 - image
    tl.store(image_ptr + offsets, image, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    width, height = 64, 32
    image = torch.randint(0, 256, (height * width * 4,), dtype=torch.uint8, device=device)
    expected = image.clone()
    rgb_mask = (torch.arange(image.numel(), device=device) % 4) != 3
    expected[rgb_mask] = 255 - expected[rgb_mask]

    solve(image, width, height)

    assert torch.equal(image, expected), (
        f"mismatch at {(image != expected).nonzero().flatten()[:8].tolist()}"
    )
    print(f"PASS [{device}] color_inversion {width}x{height}")
