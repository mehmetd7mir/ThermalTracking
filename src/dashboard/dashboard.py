"""
Real-time Dashboard for ThermalTracking
-----------------------------------------
Streamlit based dashboard for monitoring and visualization.

Features:
    - Live video feed with detections
    - Real-time statistics
    - Detection history
    - Zone visualization
    - Alert log

Run with:
    streamlit run dashboard.py
    
Author: Mehmet Demir
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import numpy as np

# try import streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not found. Install: pip install streamlit")


class DashboardState:
    """Manage dashboard state across reruns."""
    
    def __init__(self):
        self.detection_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=100)
        self.fps_history: deque = deque(maxlen=60)
        self.class_counts: Dict[str, int] = {}
        self.total_detections = 0
        self.start_time = time.time()
    
    def add_detection(self, class_name: str, confidence: float):
        """Record new detection."""
        self.detection_history.append({
            "class": class_name,
            "confidence": confidence,
            "time": datetime.now().isoformat()
        })
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.total_detections += 1
    
    def add_alert(self, level: str, message: str):
        """Record alert."""
        self.alert_history.append({
            "level": level,
            "message": message,
            "time": datetime.now().isoformat()
        })
    
    def add_fps(self, fps: float):
        """Record FPS sample."""
        self.fps_history.append(fps)
    
    def get_avg_fps(self) -> float:
        """Get average FPS."""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_uptime(self) -> str:
        """Get uptime string."""
        seconds = int(time.time() - self.start_time)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_metrics_row(state: DashboardState):
    """Create metrics row with key statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Detections",
            value=state.total_detections
        )
    
    with col2:
        st.metric(
            label="Average FPS",
            value=f"{state.get_avg_fps():.1f}"
        )
    
    with col3:
        st.metric(
            label="Active Alerts",
            value=len([a for a in state.alert_history if a["level"] == "critical"])
        )
    
    with col4:
        st.metric(
            label="Uptime",
            value=state.get_uptime()
        )


def create_class_distribution(state: DashboardState):
    """Create bar chart of class distribution."""
    if not state.class_counts:
        st.info("No detections yet")
        return
    
    import pandas as pd
    
    data = pd.DataFrame({
        "Class": list(state.class_counts.keys()),
        "Count": list(state.class_counts.values())
    })
    
    st.bar_chart(data.set_index("Class"))


def create_detection_timeline(state: DashboardState):
    """Create timeline of detections."""
    if not state.detection_history:
        st.info("No detection history")
        return
    
    import pandas as pd
    
    # get last 100 detections
    recent = list(state.detection_history)[-100:]
    
    df = pd.DataFrame(recent)
    df["time"] = pd.to_datetime(df["time"])
    
    # group by second
    df["second"] = df["time"].dt.floor("s")
    grouped = df.groupby("second").size().reset_index(name="count")
    
    st.line_chart(grouped.set_index("second"))


def create_alert_log(state: DashboardState):
    """Display alert log."""
    if not state.alert_history:
        st.info("No alerts")
        return
    
    for alert in reversed(list(state.alert_history)[-10:]):
        if alert["level"] == "critical":
            st.error(f"[{alert['time']}] {alert['message']}")
        elif alert["level"] == "warning":
            st.warning(f"[{alert['time']}] {alert['message']}")
        else:
            st.info(f"[{alert['time']}] {alert['message']}")


def run_dashboard():
    """Main dashboard function."""
    st.set_page_config(
        page_title="ThermalTracking Dashboard",
        page_icon="*",
        layout="wide"
    )
    
    st.title("ThermalTracking Dashboard")
    st.caption("Real-time monitoring and visualization")
    
    # initialize state
    if "dashboard_state" not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
    
    state = st.session_state.dashboard_state
    
    # sidebar
    with st.sidebar:
        st.header("Settings")
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05
        )
        
        show_zones = st.checkbox("Show Zones", value=True)
        show_trajectories = st.checkbox("Show Trajectories", value=True)
        
        st.divider()
        
        if st.button("Clear History"):
            state.detection_history.clear()
            state.class_counts.clear()
            state.total_detections = 0
            st.experimental_rerun()
    
    # main content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Live Feed")
        
        # video source selection
        source_type = st.radio(
            "Source",
            ["Webcam", "Video File", "RTSP Stream"],
            horizontal=True
        )
        
        if source_type == "Video File":
            uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
            if uploaded:
                st.video(uploaded)
        else:
            st.info("Configure video source to start monitoring")
        
        # placeholder for video frame
        video_placeholder = st.empty()
    
    with col_right:
        st.subheader("Statistics")
        create_metrics_row(state)
        
        st.subheader("Class Distribution")
        create_class_distribution(state)
    
    # bottom section
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Timeline")
        create_detection_timeline(state)
    
    with col2:
        st.subheader("Alert Log")
        create_alert_log(state)
    
    # add some demo data for testing
    if st.button("Add Demo Detection"):
        import random
        classes = ["drone", "bird", "plane", "helicopter"]
        state.add_detection(
            class_name=random.choice(classes),
            confidence=random.uniform(0.5, 0.95)
        )
        st.experimental_rerun()
    
    if st.button("Add Demo Alert"):
        import random
        levels = ["info", "warning", "critical"]
        messages = [
            "Drone detected in restricted area",
            "High traffic detected",
            "System running normally",
            "Connection restored"
        ]
        state.add_alert(
            level=random.choice(levels),
            message=random.choice(messages)
        )
        st.experimental_rerun()


# simple dashboard without streamlit (using gradio as fallback)
def create_gradio_dashboard():
    """Create dashboard using Gradio as fallback."""
    try:
        import gradio as gr
    except ImportError:
        print("Neither Streamlit nor Gradio available")
        return
    
    with gr.Blocks(title="ThermalTracking Dashboard") as demo:
        gr.Markdown("# ThermalTracking Dashboard")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Live Feed")
                video = gr.Video(label="Video Source")
                image_output = gr.Image(label="Detection Output")
            
            with gr.Column():
                gr.Markdown("## Statistics")
                total_det = gr.Number(label="Total Detections", value=0)
                avg_fps = gr.Number(label="Average FPS", value=0)
        
        with gr.Row():
            gr.Markdown("## Recent Detections")
            detection_log = gr.Textbox(
                label="Log",
                lines=10,
                interactive=False
            )
    
    return demo


# run
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_dashboard()
    else:
        print("Run: streamlit run dashboard.py")
        print("Or install streamlit: pip install streamlit")
