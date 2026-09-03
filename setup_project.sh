#!/bin/bash

set -e

echo "=============================================="
echo " Smart Scan Strategy for Electronic Warfare"
echo " Project Structure Setup"
echo "=============================================="

echo ""
echo "[1/4] Creating directories..."

mkdir -p simulation/{radar_model,emitter_model,channel_model,noise_model,scenarios}

mkdir -p signal_processing/{preprocessing,ms_unet1d,pulse_detection}

mkdir -p pdw/{extraction,features,validation}

mkdir -p deinterleaving/{sedcam,sem,cvr_dcm,dbscan}

mkdir -p temporal_model/{lstm,gru,emitter_representation}

mkdir -p reinforcement_learning/{environment,state,action,reward,ppo,double_dqn,training}

mkdir -p backend/{inference,scheduler,emitter_database,api}

mkdir -p frontend/{qt,dashboard,tracks,signal_view,scheduler_view}

mkdir -p datasets/{synthetic,recorded,processed}

mkdir -p models/{ms_unet1d,temporal,rl}

mkdir -p evaluation/{signal_metrics,deinterleaving_metrics,tracking_metrics,rl_metrics}

mkdir -p tests/{simulation,signal_processing,deinterleaving,temporal_model,reinforcement_learning,backend}

mkdir -p docs/{architecture,research,experiments}

mkdir -p configs

echo "[2/4] Creating Python files..."

touch main.py

touch simulation/main.py
touch simulation/radar_model/radar.py
touch simulation/emitter_model/emitter.py
touch simulation/channel_model/channel.py
touch simulation/noise_model/noise.py
touch simulation/scenarios/scenario_manager.py

touch signal_processing/pipeline.py
touch signal_processing/preprocessing/preprocess.py
touch signal_processing/ms_unet1d/model.py
touch signal_processing/ms_unet1d/dataset.py
touch signal_processing/ms_unet1d/train.py
touch signal_processing/ms_unet1d/inference.py
touch signal_processing/pulse_detection/detector.py

touch pdw/extraction/extractor.py
touch pdw/features/features.py
touch pdw/validation/validator.py

touch deinterleaving/sedcam/deinterleaver.py
touch deinterleaving/sem/sem.py
touch deinterleaving/cvr_dcm/cvr_dcm.py
touch deinterleaving/dbscan/clustering.py

touch temporal_model/lstm/model.py
touch temporal_model/lstm/train.py
touch temporal_model/gru/model.py
touch temporal_model/gru/train.py
touch temporal_model/emitter_representation/encoder.py

touch reinforcement_learning/environment/ew_environment.py
touch reinforcement_learning/state/state_space.py
touch reinforcement_learning/action/action_space.py
touch reinforcement_learning/reward/reward_function.py
touch reinforcement_learning/ppo/agent.py
touch reinforcement_learning/ppo/train.py
touch reinforcement_learning/double_dqn/agent.py
touch reinforcement_learning/double_dqn/train.py
touch reinforcement_learning/training/train.py
touch reinforcement_learning/training/evaluate.py

touch backend/inference/inference.py
touch backend/scheduler/scheduler.py
touch backend/emitter_database/database.py
touch backend/api/server.py

touch evaluation/signal_metrics/metrics.py
touch evaluation/deinterleaving_metrics/metrics.py
touch evaluation/tracking_metrics/metrics.py
touch evaluation/rl_metrics/metrics.py

touch tests/test_pipeline.py

echo "[3/4] Creating configuration files..."

touch configs/simulation.yaml
touch configs/model.yaml
touch configs/rl.yaml

echo "[4/4] Creating Python package files..."

find simulation signal_processing pdw deinterleaving \
     temporal_model reinforcement_learning backend \
     evaluation tests \
     -type d -exec touch {}/__init__.py \;

echo ""
echo "=============================================="
echo " Project structure created successfully!"
echo "=============================================="
echo ""
echo "Run:"
echo "  git status"
echo ""
echo "Then commit with:"
echo "  git add ."
echo "  git commit -m \"Initialize project architecture\""
echo "  git push"
echo ""
