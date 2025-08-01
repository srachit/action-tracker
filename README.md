# Action Tracker 🎯

A computer vision learning project focused on multi-target tracking, interactive segmentation, and object handoff detection using SAM2 (Segment Anything Model 2).

## 🎯 Project Overview

This project explores advanced computer vision techniques for tracking people and detecting interactions between them, particularly focusing on:

- **Multi-target tracking**: Simultaneously tracking both gift givers and receivers
- **Interactive segmentation**: Using SAM2 with mouse-based selection for person segmentation
- **Object handoff detection**: Identifying when objects are passed between people
- **Real-time processing**: Efficient video processing with cv2 and SAM2VideoPredictor

## 🚀 Technologies & Approach

### Core Technologies
- **SAM2 (Segment Anything Model 2)**: For interactive video segmentation
- **OpenCV (cv2)**: For video loading and processing
- **Python**: Primary development language

### Key Features
- **Interactive Selection**: Mouse-based callbacks for selecting people to track
- **Class-based Architecture**: Clean, maintainable code structure with state management
- **Multi-device Support**: CPU, CUDA, and MPS (Apple Silicon) compatibility
- **Step-by-step Learning**: Modular approach to understanding each component

## 📚 Learning Journey

This is a hands-on learning project that follows a step-by-step approach:

1. **SAM2 Basics**: Understanding SAM2's input formats and capabilities
2. **Interactive Segmentation**: Implementing mouse callbacks with predictor access
3. **Multi-target Tracking**: Tracking multiple people simultaneously
4. **Object Handoff Detection**: Detecting interactions between tracked individuals

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- SAM2 (with CPU/MPS support via PR #192)
- OpenCV
- Compatible hardware (CPU, CUDA GPU, or Apple Silicon with MPS)

### Installation
```bash
# Clone the repository
git clone https://github.com/srachit/action-tracker.git
cd action-tracker

# Install dependencies (will be added as project develops)
pip install -r requirements.txt
```

## 🎓 Learning Philosophy

This project emphasizes:
- **Learning by doing** rather than having code written automatically
- **Independent exploration** of new technologies before integration
- **Interactive development** with hands-on experimentation
- **Class-based design** for clean state management

## 🔄 Current Status

🚧 **In Development** - Currently exploring SAM2 fundamentals and building foundational understanding.

## 📝 Notes

- SAM2 now officially supports CPU and MPS devices (Apple Silicon)
- Focus on interactive, mouse-based selection approaches
- Preference for multi-target tracking over single-target approaches
- Class variables used for storing state like masks and tracking data

---

*This is a learning project focused on understanding computer vision concepts through hands-on implementation.*
