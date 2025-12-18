🚁 TinyFL-UAV: Federated Learning for Edge-Intelligent UAV Swarms

A UAV edge intelligence system that combines TinyML and Federated Learning to enable real-time, privacy-preserving AI inference directly on drone hardware. The project demonstrates collaborative learning across UAVs without sharing raw data, reducing latency and communication overhead.

🚀 Features

🧠 Lightweight TinyML CNN model for on-device inference
🤝 Federated Learning using Flower framework
🔒 Privacy-preserving training (no raw data sharing)
⚡ Low-latency edge inference with TFLite
📡 Reduced communication overhead via model compression
🌍 UAV swarm mobility and network delay simulation
🎨 Streamlit UI for real-time image classification

📁 Project Structure
TinyFL-UAV/
├── data/
│   ├── smoke/
│   └── no_smoke/
│
├── tinyml_model/
│   ├── train_cnn_pruned.py
│   ├── quantize_model.py
│   ├── convert_to_tflite.py
│   ├── model_full.h5
│   ├── model_pruned.h5
│   ├── model_quant.tflite
│   └── model_micro.tflite
│
├── federated_server/
│   ├── server_flower.py
│   ├── compression.py
│   └── adaptive_selection.py
│
├── uav_clients/
│   ├── client_flower.py
│   └── uav_config.json
│
├── simulation/
│   ├── uav_mobility_sim.py
│   ├── ns3_delay_sim.py
│   └── fl_round_visualizer.py
│
├── gazebo_world/
│   ├── uav_sensors.world
│   └── launch_uav_swarm.launch
│
├── results/
│   ├── accuracy_plot.png
│   ├── latency_plot.png
│   ├── mobility_map.png
│   └── comm_overhead.png
│
├── app.py                  # Streamlit UI
├── requirements.txt
└── README.md

🛠 Setup Instructions
Clone the Repository
git clone https://github.com/kiran280321/TinyFL-UAV.git
cd TinyFL-UAV

Install Dependencies
pip install -r requirements.txt

🧪 Train TinyML Model
cd tinyml_model
python train_cnn_pruned.py
python quantize_model.py
python convert_to_tflite.py

🌐 Run Federated Learning
Start Federated Server
cd federated_server
python server_flower.py

Start UAV Client
cd uav_clients
python client_flower.py


Each UAV trains locally and shares compressed model updates with the server.

🖥 Run Streamlit App
python -m streamlit run app.py


👉 App runs at: http://localhost:8501

📊 Results

Model Accuracy: ~85–90% after FL convergence

Latency: Significantly lower than cloud-based inference

Communication Overhead: Reduced via quantized updates

Privacy: No image data transmitted

Graphs available in the results/ folder.

🎯 Applications

🔥 Forest fire & smoke detection
🚑 Disaster response & rescue missions
🌾 Agricultural crop monitoring
🛡 Border surveillance
🏙 Smart city aerial monitoring

🧠 Skills Demonstrated

TinyML & Model Optimization
Federated Learning
Edge AI & UAV Systems
Python & TensorFlow
Simulation & Visualization
Streamlit Deployment

👨‍🎓 Authors

Made with ❤️ by
Kiran Y
