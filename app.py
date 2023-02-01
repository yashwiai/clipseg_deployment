import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image

device = torch.device("cpu")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def process_prediction(pred, image, threhsold):
    mask = Image.fromarray(np.uint8(pred * 255), "L")
    mask = mask.convert("RGB")
    mask = mask.resize(image.size)
    mask = np.array(mask)[:, :, 0]

    # normalize the mask
    mask_min = mask.min()
    mask_max = mask.max()
    mask = (mask - mask_min) / (mask_max - mask_min)

    # threshold the mask
    bmask = mask > threhsold
    # zero out values below the threshold
    mask[mask < threhsold] = 0

    return bmask, mask

def plot_heatmap(image, bmask, mask, alpha_value=0.5):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha_value, cmap="jet")

    contours, hierarchy = cv2.findContours(
        bmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = plt.Rectangle(
            (x, y), w, h, fill=False, edgecolor="yellow", linewidth=2
        )
        ax.add_patch(rect)

    ax.axis("off")
    plt.tight_layout()

    return fig

def process_image(image, text_prompt, threhsold, visual_prompt=None):
    inputs = processor(
        text=text_prompt, images=image, padding="max_length", return_tensors="pt"
    )

    # predict
    with torch.no_grad():
        if visual_prompt is not None:
            encoded_prompt = processor(images=visual_prompt, return_tensors="pt")
            outputs = model(**inputs, conditional_pixel_values=encoded_prompt.pixel_values)
        else:
            outputs = model(**inputs)

        preds = outputs.logits

    pred = torch.sigmoid(preds)
    bmask, mask = process_prediction(pred.cpu().numpy(), image, threhsold)
    segmented_image = (np.array(image) * mask[:, :, None]).astype(int)
    heatmap = plot_heatmap(image, bmask, mask)

    return heatmap, mask, segmented_image

if __name__ == "__main__":

    title = "Interactive demo: zero-shot image segmentation with CLIPSeg"
    description = "Demo for using CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. To use it, simply upload an image and add a text to mask (identify in the image), or use one of the examples below and click 'submit'. Results will show up in a few seconds."
    article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"


    with gr.Blocks() as demo:
        gr.Markdown("# CLIPSeg: Image Segmentation Using Text and Image Prompts")
        gr.Markdown(article)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil")
                input_prompt = gr.Textbox(label="Please describe what you want to identify")
                input_slider_T = gr.Slider(
                    minimum=0, maximum=1, value=0.4, label="Threshold"
                )
                btn_process = gr.Button(label="Process")

            with gr.Column():
                output_plot = gr.Plot(label="Segmentation Result")
                output_mask = gr.Image(label="Mask")
                output_image = gr.Image(label="Segmented Image")

        btn_process.click(
            process_image,
            inputs=[
                input_image,
                input_prompt,
                input_slider_T
            ],
            outputs=[output_plot, output_mask, output_image],
        )

    demo.launch(share=True)