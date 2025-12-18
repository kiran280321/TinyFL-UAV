🚁 TinyFL-UAV: Federated Learning for Edge-Intelligent UAV Swarm Systems
📌 Project Overview

TinyFL-UAV is a final-year engineering project that demonstrates how Tiny Machine Learning (TinyML) and Federated Learning (FL) can be combined to enable real-time, privacy-preserving intelligence on Unmanned Aerial Vehicles (UAVs).

The project focuses on deploying lightweight deep learning models directly on UAVs and allowing multiple UAVs to collaboratively learn using Federated Learning, without sharing raw image data. This approach reduces latency, preserves privacy, and supports deployment in remote or low-connectivity environments.

🎯 Motivation

Traditional UAV-based AI systems depend heavily on cloud servers for processing, which leads to:

High latency

Network dependency

Privacy risks

Poor performance in remote locations

This project addresses these challenges by:

Running optimized TinyML models on UAV hardware

Using Federated Learning to share only model updates

Reducing communication overhead using model compression

🧠 Key Technologies Used

TinyML – Lightweight CNN models for edge inference

Federated Learning (Flower Framework) – Distributed model training

TensorFlow & TensorFlow Lite – Model training and deployment

Model Optimization – Pruning, Quantization

Python – Core development language

Simulation Tools – UAV mobility and network delay analysis

Streamlit – User interface for real-time image classification

📂 Project Folder Structure
TinyFL-UAV/
│── README.md
│── requirements.txt
│
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
│   ├── model_micro.tflite
│   └── model_micro.cc
│
├── federated_server/
│   ├── server_flower.py
│   ├── compression.py
│   └── adaptive_selection.py
│
├── uav_clients/
│   ├── client_flower.py
│   ├── ros_image_listener.py
│   └── uav_config.json
│
├── simulation/
│   ├── ns3_delay_sim.py
│   ├── uav_mobility_sim.py
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
│   ├── comm_overhead.png
│   └── logs.txt
│
└── notebooks/
    └── full_training_pipeline.ipynb

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/TinyFL-UAV.git
cd TinyFL-UAV

2️⃣ Install Dependencies
pip install -r requirements.txt

🏋️ Model Training (TinyML)

Train and prune the CNN model:

cd tinyml_model
python train_cnn_pruned.py


Quantize the trained model:

python quantize_model.py


Convert to TensorFlow Lite:

python convert_to_tflite.py

🌐 Federated Learning Execution
Start Federated Server
cd federated_server
python server_flower.py

Start UAV Client
cd uav_clients
python client_flower.py


Each UAV trains locally and sends only compressed model updates to the server.

🖥️ Streamlit User Interface

Run the real-time image classification UI:

python -m streamlit run app.py


Features:

Upload UAV images

Real-time smoke detection

Confidence score display

📊 Results & Performance

Model Accuracy: ~85–90% after federated convergence

Latency: Reduced significantly compared to cloud-based inference

Communication Overhead: Lowered using quantized model updates

Privacy: No raw images shared between UAVs

Graphs and logs are available in the results/ folder.

🚀 Applications

Forest fire and smoke detection

Disaster response and search & rescue

Agricultural crop health monitoring

Border and perimeter surveillance

Smart city aerial monitoring

🔮 Future Scope

Deployment on real UAV hardware

Larger UAV swarms with dynamic participation

Secure aggregation and differential privacy

Reinforcement learning for autonomous navigation

Integration with IoT and ground sensors
