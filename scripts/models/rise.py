import numpy as np
import torch
import tqdm as tqdm


def generate_masks(N, s, p1, H, W):
    """
    Génère N masques aléatoires upsampled en HxW
    """
    masks = np.random.binomial(1, p1, size=(N, s, s)).astype(np.float32)

    masks = torch.tensor(masks).unsqueeze(1)  # (N,1,s,s)
    masks = torch.nn.functional.interpolate(
        masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    return masks.squeeze(1)  # (N,H,W)


def compute_rise_saliency(
    model,
    input_tensor,
    pred_class,
    N=300,
    s=8,
    p1=0.5,
    device="cpu"
):
    _, _, H, W = input_tensor.shape

    masks = generate_masks(N, s, p1, H, W).to(device)
    saliency = torch.zeros(H, W, device=device)

    model.eval()

    with torch.no_grad():
        for i in tqdm.tqdm(range(N), desc="RISE"):

            mask = masks[i:i+1]                     # (1,H,W)
            masked_input = input_tensor * mask.unsqueeze(1)  # (1,3,H,W)

            output = model(masked_input)
            prob = torch.softmax(output, dim=1)[0, pred_class]

            saliency += prob * mask.squeeze(0)

    # --- NORMALISATION FINALE ---
    saliency = saliency / N

    saliency = saliency.cpu().numpy()

    saliency = (
        (saliency - saliency.min())
        / (saliency.max() - saliency.min() + 1e-8)
    )

    return saliency.astype(np.float32)
