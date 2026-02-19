import argparse
from pathlib import Path

from PIL import Image

from xaif.scripts.tools_for_data.data import image_to_tensor, load_image
from xaif.scripts.models.rise import compute_rise_saliency
from xaif.scripts.models.reset import get_resnet50, predict_class
from xaif.scripts.tools_for_data.visualize import save_overlay_from_pil


def main():
    parser = argparse.ArgumentParser(description="Run RISE on one image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--out", default="outputs/rise", help="Output directory.")
    parser.add_argument("--num-masks", type=int, default=300, help="Number of random masks.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = get_resnet50(device=args.device)
    input_tensor = image_to_tensor(args.image, image_size=224, device=args.device)
    pred_class = predict_class(model, input_tensor)

    saliency = compute_rise_saliency(
        model=model,
        input_tensor=input_tensor,
        pred_class=pred_class,
        num_masks=args.num_masks,
        s=8,
        p1=0.5,
        batch_size=50,
    )

    img = load_image(args.image)
    save_overlay_from_pil(img, saliency, out_dir / "rise_overlay.jpg")
    img.resize((224, 224)).save(out_dir / "input.jpg")
    Image.fromarray((saliency * 255).astype("uint8")).save(out_dir / "rise_map.jpg")

    print(f"Saved RISE outputs to: {out_dir}")
    print(f"Predicted class index: {pred_class}")


if __name__ == "__main__":
    main()

