# Smart Scan Strategy for Electronic Warfare

AI-driven Electronic Warfare signal processing and
intelligent sensor scheduling system.

## System Architecture

EW Simulation
        ↓
RF / IQ Signals
        ↓
MS UNet1D
        ↓
Pulse Segmentation
        ↓
PDW Extraction
TOA / PRI / CF / PW / DOA / PA
        ↓
SEDCAM Deinterleaving
SEM → CVR DCM → DBSCAN
        ↓
Emitter Representation
        ↓
LSTM / GRU Temporal State
        ↓
Reinforcement Learning
PPO / Double DQN
        ↓
Intelligent Sensor Scheduling
        ↓
EW Environment
        ↓
New Observation
        ↺

## Main Components

### Signal Processing
- RF / IQ signal preprocessing
- MS UNet1D pulse segmentation
- Pulse detection

### PDW Extraction
- TOA
- PRI
- CF
- PW
- DOA
- PA

### Deinterleaving
- SEDCAM
- SEM
- CVR DCM
- DBSCAN

### Temporal Modeling
- LSTM
- GRU
- Emitter representation

### Reinforcement Learning
- PPO
- Double DQN
- Dynamic sensor scheduling
- Resource-aware decision making

### Frontend
- C++
- Qt / Qt Quick

### Backend
- Python
- Model inference
- Scheduler
- Emitter database

## Technology Stack

Python
C++
Qt
PyTorch
Stable Baselines3
Gymnasium
NumPy
SciPy
scikit-learn
YAML

## Project Status

Under active development.

