"""
ThermalTracking - Gradio Web Demo
==================================
Interaktif web arayüzü ile thermal görüntü tespiti.

Özellikler:
- Görüntü/Video yükleme
- Real-time detection
- Tehdit seviyesi gösterimi
- İstatistik dashboard

Kullanım:
    python app.py
    # Browser'da http://localhost:7860 aç

Hugging Face Deployment:
    1. https://huggingface.co/spaces adresinden yeni Space oluştur
    2. Bu dosyayı app.py olarak yükle
    3. requirements.txt ekle
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not found. Install with: pip install gradio")

try:
    from ultralytics import YOLO
    import cv2
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Global model (lazy loading)
_model = None
_model_path = None


def load_model(weights: str = "best.pt"):
    """Model'i lazy load et."""
    global _model, _model_path
    
    if _model is None or _model_path != weights:
        if Path(weights).exists():
            _model = YOLO(weights)
            _model_path = weights
            print(f"✅ Model yüklendi: {weights}")
        else:
            print(f"⚠️ Model bulunamadı: {weights}")
            _model = YOLO("yolov8n.pt")  # Fallback to nano
            _model_path = "yolov8n.pt"
    
    return _model


def process_image(
    image: np.ndarray,
    confidence: float = 0.25,
    model_path: str = "best.pt"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tek bir görüntüyü işle.
    
    Args:
        image: Input image
        confidence: Confidence threshold
        model_path: Model weights
    
    Returns:
        (annotated_image, statistics)
    """
    model = load_model(model_path)
    
    # Inference
    results = model.predict(image, conf=confidence, verbose=False)
    
    # Annotate
    annotated = results[0].plot()
    
    # İstatistikler
    stats = {
        "total_detections": 0,
        "classes": {},
        "confidences": [],
        "threat_levels": {"high": 0, "medium": 0, "low": 0}
    }
    
    class_names = {0: 'bird', 1: 'drone', 2: 'helicopter', 3: 'plane'}
    threat_scores = {'drone': 3, 'helicopter': 2, 'plane': 2, 'bird': 1}
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = class_names.get(cls_id, 'unknown')
            
            stats["total_detections"] += 1
            stats["classes"][cls_name] = stats["classes"].get(cls_name, 0) + 1
            stats["confidences"].append(conf)
            
            # Basit threat level
            threat_score = threat_scores.get(cls_name, 1)
            if threat_score >= 3:
                stats["threat_levels"]["high"] += 1
            elif threat_score >= 2:
                stats["threat_levels"]["medium"] += 1
            else:
                stats["threat_levels"]["low"] += 1
    
    return annotated, stats


def process_video(
    video_path: str,
    confidence: float = 0.25,
    model_path: str = "best.pt",
    max_frames: int = 300  # Limit for demo
) -> Tuple[str, Dict[str, Any]]:
    """
    Video'yu işle.
    
    Args:
        video_path: Input video path
        confidence: Confidence threshold
        model_path: Model weights
        max_frames: Maximum frames to process
    
    Returns:
        (output_video_path, statistics)
    """
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, {"error": "Video açılamadı"}
    
    # Video özellikleri
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video
    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # İstatistikler
    total_detections = 0
    class_counts = {}
    frame_count = 0
    
    class_names = {0: 'bird', 1: 'drone', 2: 'helicopter', 3: 'plane'}
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        frame_count += 1
        
        # Inference
        results = model.predict(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()
        
        # Count detections
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = class_names.get(cls_id, 'unknown')
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_detections += 1
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    stats = {
        "total_frames": frame_count,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / max(1, frame_count),
        "class_distribution": class_counts,
        "video_fps": fps
    }
    
    return output_path, stats


def create_stats_html(stats: Dict[str, Any]) -> str:
    """İstatistikleri HTML olarak formatla."""
    if "error" in stats:
        return f"<div style='color: red;'>❌ {stats['error']}</div>"
    
    html = """
    <div style='font-family: Arial; padding: 10px;'>
        <h3>📊 İstatistikler</h3>
        <table style='width: 100%; border-collapse: collapse;'>
    """
    
    for key, value in stats.items():
        if key == "classes" or key == "class_distribution":
            value_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
        elif key == "threat_levels":
            value_str = f"🔴 High: {value['high']}, 🟡 Medium: {value['medium']}, 🟢 Low: {value['low']}"
        elif isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        html += f"""
        <tr style='border-bottom: 1px solid #ddd;'>
            <td style='padding: 8px; font-weight: bold;'>{key.replace('_', ' ').title()}</td>
            <td style='padding: 8px;'>{value_str}</td>
        </tr>
        """
    
    html += "</table></div>"
    return html


def image_interface(image, confidence):
    """Gradio image interface handler."""
    if image is None:
        return None, "<p>Lütfen bir görüntü yükleyin</p>"
    
    annotated, stats = process_image(image, confidence)
    stats_html = create_stats_html(stats)
    
    return annotated, stats_html


def video_interface(video, confidence):
    """Gradio video interface handler."""
    if video is None:
        return None, "<p>Lütfen bir video yükleyin</p>"
    
    output_path, stats = process_video(video, confidence)
    stats_html = create_stats_html(stats)
    
    return output_path, stats_html


def create_demo() -> gr.Blocks:
    """Gradio demo oluştur."""
    
    with gr.Blocks(
        title="🔥 ThermalTracking",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .gr-button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔥 ThermalTracking
        
        **YOLOv8-based Detection of Aerial Targets in Thermal Imagery**
        
        Thermal görüntülerde drone, uçak, helikopter ve kuş tespiti yapın.
        
        ---
        """)
        
        with gr.Tabs():
            # Image Tab
            with gr.TabItem("🖼️ Görüntü"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="Thermal Görüntü Yükle", type="numpy")
                        img_confidence = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        img_btn = gr.Button("🔍 Tespit Et", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(label="Sonuç")
                        img_stats = gr.HTML(label="İstatistikler")
                
                img_btn.click(
                    fn=image_interface,
                    inputs=[img_input, img_confidence],
                    outputs=[img_output, img_stats]
                )
            
            # Video Tab
            with gr.TabItem("🎬 Video"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Thermal Video Yükle")
                        vid_confidence = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        vid_btn = gr.Button("🔍 İşle", variant="primary")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="Sonuç")
                        vid_stats = gr.HTML(label="İstatistikler")
                
                vid_btn.click(
                    fn=video_interface,
                    inputs=[vid_input, vid_confidence],
                    outputs=[vid_output, vid_stats]
                )
            
            # About Tab
            with gr.TabItem("ℹ️ Hakkında"):
                gr.Markdown("""
                ## Proje Hakkında
                
                **ThermalTracking**, thermal (kızılötesi) görüntülerde hava araçlarını
                tespit etmek için geliştirilmiş bir yapay zeka sistemidir.
                
                ### Tespit Edilen Sınıflar
                
                | Sınıf | Açıklama | Tehdit |
                |-------|----------|--------|
                | 🛸 Drone | İnsansız hava aracı | 🔴 Yüksek |
                | 🚁 Helicopter | Helikopter | 🟡 Orta |
                | ✈️ Plane | Uçak | 🟡 Orta |
                | 🐦 Bird | Kuş | 🟢 Düşük |
                
                ### Teknik Detaylar
                
                - **Model**: YOLOv8m (25.8M parameters)
                - **Dataset**: 80,000+ augmented thermal images
                - **mAP50**: 73.9%
                
                ### Geliştirici
                
                **Mehmet Demir** - [GitHub](https://github.com/mehmetd7mir)
                """)
        
        gr.Markdown("""
        ---
        <p style='text-align: center; color: gray;'>
            Built with ❤️ for Defense & Aerospace Applications
        </p>
        """)
    
    return demo


# Main
if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("❌ Gradio yüklü değil!")
        print("   pip install gradio")
        exit(1)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Public link için True yap
        show_error=True
    )
