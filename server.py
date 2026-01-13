import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import io
import base64
from model import Net  # 确保目录下有 model.py

app = Flask(__name__)
CORS(app)

# --- 全局配置 ---
DEVICE = torch.device("cpu") # 推理通常用 CPU 足够
models = {} # 模型注册表

# --- 1. 加载模型 (日志保持中文) ---
def load_models():
    print("📦 正在初始化模型仓库...")
    
    # === 模型 1: 数字识别 ===
    try:
        net_digit = Net().to(DEVICE)
        net_digit.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
        net_digit.eval()
        models['digit'] = net_digit
        print("   ✅ [digit] 数字识别模型加载完毕")
    except Exception as e:
        print(f"   ⚠️ [digit] 加载失败 (请确保目录下有 mnist_cnn.pth): {e}")

    # === 模型 2: (预留) 字母识别 ===
    # try:
    #     net_letter = LetterNet().to(DEVICE)
    #     net_letter.load_state_dict(torch.load("emnist_letters.pth"))
    #     models['letter'] = net_letter
    # except: pass

    print(f"🎉 服务启动完成，当前可用模型: {list(models.keys())}")

# --- 2. 图像预处理 ---
def process_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0).to(DEVICE)

# --- 3. 动态路由接口 ---
@app.route('/predict/<model_name>', methods=['POST'])
def predict_router(model_name):
    # 1. 检查模型是否存在
    if model_name not in models:
        # 返回给前端的错误信息用英文
        return jsonify({'error': f"Model '{model_name}' not deployed or loaded"}), 404
    
    # 2. 获取数据
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    try:
        # 3. 预处理
        tensor = process_image(data['image'])
        
        # 4. 推理
        selected_model = models[model_name]
        with torch.no_grad():
            output = selected_model(tensor)
            pred_index = output.argmax(dim=1).item()
        
        # 5. 返回结果
        return jsonify({
            'model': model_name,
            'prediction': pred_index,
            'status': 'success'
        })

    except Exception as e:
        # 捕获异常也返回英文给前端
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models() 
    app.run(host='0.0.0.0', port=5000, debug=True)