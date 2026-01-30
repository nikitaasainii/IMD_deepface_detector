import os
import sys
import cv2
import torch
import gradio as gr

# ==========================================
# 1. THE PATH BRIDGE (Fixes ModuleNotFoundError)
# ==========================================
# This adds the parent directory to sys.path so 'deepfake_detector' is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your specific repository modules
try:
    from deepfake_detector.models import DeepFakeDetector
    from deepfake_detector.data.transforms import get_val_transforms
    print("✅ Successfully imported repository modules.")
except ImportError as e:
    print(f"❌ Still can't find modules: {e}")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
def load_model():
    # Update this to match your specific model and checkpoint path
    checkpoint_path = "outputs/checkpoints/best_model.pth" 
    model = DeepFakeDetector(model_name='efficientnet-b1')
    
    if os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
        model.eval()
        return model
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}")
        return None

model = load_model()
transform = get_val_transforms(image_size=240)

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
def predict(input_img):
    if input_img is None or model is None:
        return "System Not Ready", "0%"

    # Convert to RGB and apply your repo's transforms
    image_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image=image_rgb)['image'].unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        # Adjust these indices based on how you labeled your data (0=Real, 1=Fake?)
        fake_score = probabilities[1].item()
        is_fake = fake_score > 0.5
        
    label = "🔴 DEEPFAKE" if is_fake else "🟢 REAL"
    confidence = f"{max(probabilities).item():.2%}"
    
    return label, confidence

# ==========================================
# 4. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Deepfake Detection Interface")
    
    with gr.Row():
        with gr.Column():
            input_data = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
            btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            out_label = gr.Textbox(label="Classification")
            out_conf = gr.Textbox(label="Confidence Score")

    btn.click(fn=predict, inputs=input_data, outputs=[out_label, out_conf])

if __name__ == "__main__":
    demo.launch()