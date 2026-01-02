# Husky Dashboard: ROS Sensor Data Poisoning Detection

**Course:** CS 4366 - Senior Capstone Project - Fall 2025  
**Author:** Max Heitzman  
**Project:** Robot System Call Analysis Platform - GPS Anomaly Detection Component

## 📋 Project Overview

This is a web-based dashboard for detecting and validating sensor data poisoning attacks on ROS-based autonomous vehicles. Built as part of the Robot System Call Analysis Platform capstone project, this system monitors GPS, IMU, and odometry data from the UMD Husky robot to detect anomalies and potential security threats in real-time.

## 🎯 Key Features

### 1. Live Dashboard
- **Real-time Processing Metrics:** Monitor system performance and throughput
- **GPS Trajectory Visualization:** Interactive map showing robot path with poisoned segments highlighted
- **Sensor Trust Score Indicators:** Visual trust metrics for each sensor
- **Anomaly Detection Log:** Real-time log of detected anomalies

### 2. Poison Injection System
Inject controlled poisoning attacks during bag file processing to test detection capabilities:

- **GPS Jump** - Sudden position jump (simulates GPS spoofing)
- **GPS Drift** - Gradual position drift (simulates GPS jamming)
- **GPS Freeze** - Frozen GPS readings (simulates sensor failure)
- **IMU Noise** - Add noise to IMU data (simulates sensor degradation)
- **IMU Bias** - Add constant bias (simulates calibration errors)
- **Odometry Scaling** - Scale odometry values (simulates wheel slip)

### 3. Preset Attack Scenarios
Pre-configured attack scenarios for testing:

- `quick_test` - Two GPS jumps at 2s and 5s
- `sustained_drift` - 5 m/s GPS drift for 5 seconds
- `sensor_freeze` - GPS freeze for 3 seconds
- `imu_attack` - IMU noise + bias attacks
- `multi_sensor` - Combined GPS, IMU, and Odometry attacks

### 4. Bag File Support
- **ROS1**: `.bag` files
- **ROS2**: `.zip` files containing `.db3` and `metadata.yaml`

**Note:** All bag files are assumed to be clean/unpoisoned. Poisoning is injected during processing to simulate attacks.

## 🏗️ Technical Architecture

### System Components

1. **Flask Backend (`app.py`)**
   - RESTful API endpoints
   - File upload handling
   - Real-time processing management
   - WebSocket support for live updates

2. **ROS Bag Parser (`backend.py`)**
   - ROS1 and ROS2 bag file parsing
   - Sensor data extraction (GPS, IMU, Odometry)
   - Real-time data streaming
   - Anomaly detection integration

3. **Poison Injector (`poison_injector.py`)**
   - In-memory data manipulation
   - Multiple attack types
   - Preset scenario management
   - Timestamp-based injection

4. **GPS Prediction Model (`gps_prediction_model.py`)**
   - LSTM-based GPS anomaly detection
   - Pre-trained model with 5.0s prediction window
   - Real-time inference
   - Confidence scoring

5. **Dashboard UI (`advanced_dashboard.html`)**
   - Interactive web interface
   - Real-time visualization
   - Multi-tab interface
   - Responsive design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- ROS1 or ROS2 (for bag file generation)
- Flask and dependencies

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python3 app.py

# Open in browser
http://localhost:5000
```

### Usage

#### Test with Synthetic Data
1. Open dashboard at `http://localhost:5000`
2. Click **"🧪 Synthetic"** to run quick test
3. Watch trajectory and anomalies appear

#### Process Bag File with Poison Injection
1. Go to **"💉 Inject Poison"** tab
2. Select a preset (e.g., `quick_test`) or configure custom attack
3. Click **"Apply Preset"**
4. Go back to **"📊 Live Dashboard"** tab
5. Upload your `.bag` or `.zip` file
6. Click **"▶️ Start"**
7. Watch the poisoned trajectory - red segments show where poison was active

#### Validate Detection
1. After processing completes
2. Go to **"✓ Validate"** tab
3. Click **"Run Validation"**
4. Enter processor ID (e.g., `processor_0`)
5. Review detection accuracy metrics

## 📁 Project Structure

```
husky-dashboard/
├── app.py                    # Flask server and API endpoints
├── backend.py                # ROS bag parsing & anomaly detection
├── poison_injector.py        # Poison injection system
├── gps_prediction_model.py  # LSTM model for GPS anomaly detection
├── advanced_dashboard.html   # Main dashboard UI
├── requirements.txt          # Python dependencies
├── gps_models/              # Pre-trained LSTM models
│   ├── gps_model_lstm_5.0s.pt
│   ├── input_scaler_5.0s.pkl
│   ├── output_scaler_5.0s.pkl
│   └── metadata_5.0s.pkl
├── uploads/                  # Uploaded bag files (created at runtime)
└── README.md                 # This file
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload bag file |
| `/api/process/start` | POST | Start processing |
| `/api/process/<id>/status` | GET | Get processing status |
| `/api/process/<id>/stop` | POST | Stop processing |
| `/api/poison/presets` | GET | List attack presets |
| `/api/poison/inject` | POST | Inject poison |
| `/api/poison/clear` | POST | Clear poisons |
| `/api/poison/status` | GET | Get poison status |
| `/api/poison/validate` | POST | Validate detection |

## 💡 Technical Implementation Details

### GPS Anomaly Detection

The system uses an LSTM (Long Short-Term Memory) neural network to detect GPS anomalies:

- **Model Architecture:** LSTM with 5.0 second prediction window
- **Input Features:** GPS position, velocity, acceleration
- **Output:** Anomaly probability and confidence score
- **Training Data:** DoD SAFE dataset from UMD Husky robot

### Poison Injection Mechanism

Poison injection happens in-memory during bag file processing:

1. **Data Loading:** ROS bag file is parsed and sensor data extracted
2. **Poison Application:** Selected attack types are applied at specified timestamps
3. **Data Manipulation:** Original sensor values are modified according to attack type
4. **Detection Testing:** Modified data is passed through anomaly detection
5. **Visualization:** Results are displayed with poisoned segments highlighted

### Real-Time Processing

The system processes ROS bag files in real-time:

- **Streaming:** Data is processed as it's extracted from bag files
- **Live Updates:** Dashboard updates in real-time via WebSocket
- **Multi-threading:** Processing runs in separate threads to avoid blocking
- **Status Tracking:** Processing status is tracked and reported via API

## 🔧 Areas for Future Enhancement

### 1. Enhanced Visualization
- **Live Sensor Readings:** Display actual IMU, GPS, and Odometry values
- **Time-Series Charts:** Historical data visualization
- **Interactive Maps:** Zoom/pan on trajectory visualization
- **Heatmaps:** Anomaly density visualization

### 2. Improved Poison Injection
- **Visual Markers:** Show exact injection timestamps on trajectory
- **Before/After Comparison:** Side-by-side view of clean vs poisoned data
- **Injection Timeline:** Visual timeline of all injections
- **Real-Time Injection:** Inject poison during live data collection

### 3. Advanced Validation
- **Automatic Processor ID Detection:** No manual input required
- **Detailed Reports:** Export validation results
- **Statistical Analysis:** Comprehensive metrics and analysis
- **Historical Tracking:** Track validation results over time

### 4. Performance Optimization
- **Parallel Processing:** Process multiple bag files simultaneously
- **Caching:** Cache processed data for faster re-analysis
- **Database Integration:** Store results in database for historical analysis
- **Cloud Deployment:** Deploy to cloud for scalability

## 🎓 Capstone Project Context

This dashboard is part of the **Robot System Call Analysis Platform** capstone project, which focuses on:

- **System Call Monitoring:** Analyzing 11,200+ system calls per second from UMD Husky robot
- **Anomaly Detection:** Using LSTM neural networks for pattern recognition (94% accuracy)
- **Security Validation:** Detecting threats and performance issues in autonomous systems
- **Real-Time Monitoring:** Live dashboard for operational monitoring

### Integration with Main Platform

This GPS anomaly detection component integrates with the main platform to provide:

- Sensor-level anomaly detection
- Cross-sensor validation
- Real-time threat detection
- Historical analysis capabilities

## 📚 Dependencies

- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **rosbags** - ROS bag file parsing
- **PyTorch** - Deep learning framework (for LSTM model)

## 🔗 Related Resources

- [Original Repository](https://github.com/joMusangu/husky-dashboard/tree/feature/gps-anomaly-detection)
- [DoD SAFE Dataset](https://www.dodsafe.org/)
- [UMD Husky Robot](https://www.clearpathrobotics.com/husky-unmanned-ground-vehicle-robot/)
- [ROS Documentation](https://www.ros.org/)

## 👤 Author

**Max Heitzman**

---

*Capstone Project - CS 4366 - Fall 2025*
