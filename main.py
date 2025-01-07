import os
import random
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344
SAVE_DIR = "/content/images"
MODEL_DIR = "/content/models"
LORA_DIR = "/content/lora"

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load a model
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return None, f"Model {model_name} not found in {MODEL_DIR}."
    
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path, use_safetensors=True, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe, f"Switched to model: {model_name}"

# Preload a default model
DEFAULT_MODEL_NAME = "default.safetensors"
pipe, status_message = load_model(DEFAULT_MODEL_NAME)
if pipe is None:
    raise FileNotFoundError(f"Default model {DEFAULT_MODEL_NAME} not found in {MODEL_DIR}.")
print("\033[1;32mDefault model loaded successfully!\033[0m")

def get_available_models():
    """Returns a list of available models in the MODEL_DIR."""
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")]

def get_available_lora_models():
    """Returns a list of available LoRA models in the LORA_DIR."""
    return [f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors")]

def switch_model(selected_model):
    global pipe
    pipe, message = load_model(selected_model)
    return message

def load_lora_model(lora_name):
    if not lora_name:  # Check if lora_name is None or empty
        pipe.load_lora_weights(None)  # Remove the LoRA weights if None is selected
        return "No LoRA model selected. Reverted to base model."

    lora_path = os.path.join(LORA_DIR, lora_name)
    if not os.path.exists(lora_path):
        return f"LoRA model {lora_name} not found in {LORA_DIR}."
    
    pipe.load_lora_weights(lora_path)
    return f"Switched to LoRA model: {lora_name}"

def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if seed == -1:  # -1 indicates random seed
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator,
    ).images[0]
    
    i = 1
    while os.path.exists(os.path.join(SAVE_DIR, f"image_{i}.png")):
        i += 1

    image_path = os.path.join(SAVE_DIR, f"image_{i}.png")
    image.save(image_path)
    return image_path

# UI setup
css = """
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
footer {
    display: none !important;
}
"""

examples = [
    "a cat",
    "a cat in the hat",
    "a cat in the cowboy hat",
]

with gr.Blocks(css=css, theme='ParityError/Interstellar') as app:
    with gr.Row(elem_id="col-container"):
        # Left Column
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("""
            # Stable Diffusion with Model and LoRA Switching
            """)
            
            with gr.Row():
                available_models = gr.Dropdown(
                    choices=get_available_models(), 
                    label="Select Model", 
                    value=DEFAULT_MODEL_NAME
                )
                switch_button = gr.Button("Switch Model")
            
            model_status = gr.Text(label="Model Status", interactive=False)
            
            with gr.Row():
                available_lora_models = gr.Dropdown(
                    choices=["None"] + get_available_lora_models(),  # Add "None" option for clearing LoRA
                    label="Select LoRA Model",
                    value=None  # Default to None
                )
                lora_switch_button = gr.Button("Switch LoRA Model")
            
            lora_status = gr.Text(label="LoRA Status", interactive=False)
            
            with gr.Accordion("⚙️ Settings", open=False):
                negative_prompt = gr.Text(
                    label="Negative prompt", 
                    placeholder="Enter a negative prompt",
                    lines=3, 
                    value='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, bad anatomy,bad hands,cropped,poorly drawn hands,out of focus,monochrome,symbol,text,logo,lowres,signature,(worst quality, low quality:1.5), 3d, (3D:1.2). (bad anatomy), muscle, nipples'
                )
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=20.0, step=0.1, value=10.0)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=35)
        
        # Right Column
        with gr.Column(scale=2):
            prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=7,
                             placeholder="Enter your prompt", container=False, scale=4)
            run_button = gr.Button("🚀 Generate", variant="primary")
            result = gr.Image(label="Result", show_label=False)
            gr.Examples(examples=examples, inputs=[prompt])

    # Button interactions
    switch_button.click(
        fn=switch_model,
        inputs=[available_models],
        outputs=model_status,
    )

    lora_switch_button.click(
        fn=lambda lora: load_lora_model(lora),
        inputs=[available_lora_models],
        outputs=lora_status,
    )

    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=result,
    )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
