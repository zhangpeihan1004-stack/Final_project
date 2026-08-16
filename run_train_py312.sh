#!/bin/bash
#SBATCH --job-name=iuxray_c6_final_e4
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --output=iuxray_c6_final_e4_%j.out
#SBATCH --error=iuxray_c6_final_e4_%j.err

set -euo pipefail

PROJECT_DIR="/scr/user/zhangpeihan1004/final_colab/R2Gen-Finetuning-main"
ENV_DIR="/scr/user/zhangpeihan1004/final_colab/iuxray_py312_aw3"
PYTHON_BIN="${ENV_DIR}/bin/python"

# 支持有 .pth 和无 .pth 两种文件名
CHECKPOINT_WITH_EXT="${PROJECT_DIR}/results/checkpoint_epoch_4_final.pth"
CHECKPOINT_WITHOUT_EXT="${PROJECT_DIR}/results/checkpoint_epoch_4_final"

if [[ -f "${CHECKPOINT_WITH_EXT}" ]]; then
    PRETRAINED_FINAL_CHECKPOINT="${CHECKPOINT_WITH_EXT}"
elif [[ -f "${CHECKPOINT_WITHOUT_EXT}" ]]; then
    PRETRAINED_FINAL_CHECKPOINT="${CHECKPOINT_WITHOUT_EXT}"
else
    echo "ERROR: FinalModel checkpoint not found."
    echo "Checked: ${CHECKPOINT_WITH_EXT}"
    echo "Checked: ${CHECKPOINT_WITHOUT_EXT}"
    exit 1
fi

IMAGE_DIR="data/iu_xray/images/"
ANN_PATH="data/iu_xray/annotation.json"
VOCAB_PATH="data/iu_xray/vocab8.pkl"

# 使用新的目录，防止覆盖原 FinalModel 和以前的 Clinical6
SAVE_DIR="results/iu_xray_clinical6_from_final_epoch4"
RECORD_DIR="records/clinical6_from_final_epoch4"
EXPORT_BLEU_DIR="test_outputs/clinical6_from_final_epoch4/bleu_best"
EXPORT_CLINICAL_DIR="test_outputs/clinical6_from_final_epoch4/clinical_best"

BLEU_CHECKPOINT="${SAVE_DIR}/model_best_bleu.pth"
CLINICAL_CHECKPOINT="${SAVE_DIR}/model_best_clinical.pth"

module purge
module load miniconda/24.11.1

# 防止旧 Python 3.10 环境污染
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "Initial FinalModel: ${PRETRAINED_FINAL_CHECKPOINT}"
echo "Checkpoint size: $(stat -c '%s bytes' "${PRETRAINED_FINAL_CHECKPOINT}")"
echo "Save directory: ${SAVE_DIR}"
echo "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python environment not found: ${PYTHON_BIN}"
    exit 1
fi

required_paths=(
    "main.py"
    "export_test_results.py"
    "models/r2gen.py"
    "modules/abnormality.py"
    "modules/dataloaders.py"
    "modules/datasets.py"
    "modules/loss.py"
    "modules/optimizers.py"
    "modules/trainer.py"
    "modules/visual_extractor.py"
    "${ANN_PATH}"
    "${VOCAB_PATH}"
    "${IMAGE_DIR}"
)

for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: Required path not found: ${required_path}"
        exit 1
    fi
done

echo "Checking Python environment..."

"${PYTHON_BIN}" -c "
import sys
import torch
import torchvision
import timm
import numpy
import pandas
from PIL import Image

print('Python:', sys.version)
print('Python executable:', sys.executable)
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('timm:', timm.__version__)
print('NumPy:', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable inside the GPU job')

print('GPU:', torch.cuda.get_device_name(0))
"

nvidia-smi

echo "Checking FinalModel checkpoint structure..."

"${PYTHON_BIN}" - "${PRETRAINED_FINAL_CHECKPOINT}" <<'PY'
import os
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

state = checkpoint
if isinstance(checkpoint, dict):
    for name in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(name), dict):
            state = checkpoint[name]
            break

if not isinstance(state, dict):
    raise RuntimeError("Checkpoint does not contain a valid state dictionary")

keys = list(state.keys())

has_swin = any("swin_model" in key for key in keys)
has_mfa = any("proj_stage3" in key for key in keys)
has_resnet = any("visual_extractor.model" in key for key in keys)
has_pathology = any("pathology_classifier" in key for key in keys)

print("Checkpoint:", path)
print("Size:", os.path.getsize(path), "bytes")
print("Epoch:", checkpoint.get("epoch") if isinstance(checkpoint, dict) else None)
print(
    "Monitor best:",
    checkpoint.get("monitor_best") if isinstance(checkpoint, dict) else None
)
print("Number of parameters:", len(keys))
print("HAS_SWIN:", has_swin)
print("HAS_MFA:", has_mfa)
print("HAS_RESNET:", has_resnet)
print("HAS_PATHOLOGY_CLASSIFIER:", has_pathology)

swin_keys = [key for key in keys if "swin_model" in key]
print("First Swin keys:")
print("\n".join(swin_keys[:10]) if swin_keys else "NONE")

if not has_swin:
    raise RuntimeError("This checkpoint does not contain a Swin backbone")

if not has_mfa:
    raise RuntimeError("This checkpoint does not contain the expected MFA layers")

if has_resnet:
    raise RuntimeError("This appears to be a ResNet checkpoint, not FinalModel")

print("Basic FinalModel checkpoint check passed")
PY

mkdir -p \
    "${SAVE_DIR}" \
    "${RECORD_DIR}" \
    "${EXPORT_BLEU_DIR}" \
    "${EXPORT_CLINICAL_DIR}"

echo "Starting Clinical6 training from FinalModel epoch 4..."

srun "${PYTHON_BIN}" -u main.py \
    --dataset_name iu_xray \
    --image_dir "${IMAGE_DIR}" \
    --ann_path "${ANN_PATH}" \
    --vocab_path "${VOCAB_PATH}" \
    --visual_extractor swin_base_mfa \
    --d_vf 512 \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 40 \
    --seed 9233 \
    --optim AdamW \
    --lr_ve 5e-5 \
    --lr_ed 3e-4 \
    --lr_pathology 1e-4 \
    --weight_decay 2e-3 \
    --lr_scheduler StepLR \
    --step_size 4 \
    --gamma 0.8 \
    --lambda_cl 0.05 \
    --focal_gamma 2.0 \
    --cl_temperature 0.07 \
    --abnormal_weight 1.0 \
    --pathology_token_weight 1.0 \
    --lambda_pathology 0.2 \
    --pathology_dropout 0.2 \
    --view_fusion_dropout 0.2 \
    --pathology_threshold 0.5 \
    --threshold_search_min 0.1 \
    --threshold_search_max 0.9 \
    --threshold_search_step 0.05 \
    --balanced_disease_sampling \
    --disease_sampling_max_weight 3.0 \
    --pathology_pos_weight_cap 5.0 \
    --pathology_loss_type asymmetric \
    --asymmetric_gamma_neg 4.0 \
    --asymmetric_gamma_pos 1.0 \
    --asymmetric_clip 0.05 \
    --pretrained_final_checkpoint "${PRETRAINED_FINAL_CHECKPOINT}" \
    --freeze_backbone_epochs 3 \
    --monitor_mode max \
    --monitor_metric BLEU_4 \
    --early_stop 40 \
    --beam_size 3 \
    --block_trigrams 1 \
    --save_period 1 \
    --save_dir "${SAVE_DIR}" \
    --record_dir "${RECORD_DIR}"

if [[ ! -f "${BLEU_CHECKPOINT}" ]]; then
    echo "ERROR: BLEU-best checkpoint was not created:"
    echo "${BLEU_CHECKPOINT}"
    exit 1
fi

if [[ ! -f "${CLINICAL_CHECKPOINT}" ]]; then
    echo "ERROR: Clinical-best checkpoint was not created:"
    echo "${CLINICAL_CHECKPOINT}"
    exit 1
fi

echo "Exporting BLEU-selected model test results..."

srun "${PYTHON_BIN}" -u export_test_results.py \
    --checkpoint "${BLEU_CHECKPOINT}" \
    --device cuda \
    --export_dir "${EXPORT_BLEU_DIR}" \
    --dataset_name iu_xray \
    --image_dir "${IMAGE_DIR}" \
    --ann_path "${ANN_PATH}" \
    --vocab_path "${VOCAB_PATH}" \
    --visual_extractor swin_base_mfa \
    --d_vf 512 \
    --batch_size 32 \
    --num_workers 8 \
    --beam_size 3 \
    --block_trigrams 1

echo "Exporting clinical-F1-selected model test results..."

srun "${PYTHON_BIN}" -u export_test_results.py \
    --checkpoint "${CLINICAL_CHECKPOINT}" \
    --device cuda \
    --export_dir "${EXPORT_CLINICAL_DIR}" \
    --dataset_name iu_xray \
    --image_dir "${IMAGE_DIR}" \
    --ann_path "${ANN_PATH}" \
    --vocab_path "${VOCAB_PATH}" \
    --visual_extractor swin_base_mfa \
    --d_vf 512 \
    --batch_size 32 \
    --num_workers 8 \
    --beam_size 3 \
    --block_trigrams 1

echo "============================================================"
echo "Training completed"
echo "BLEU-best model: ${BLEU_CHECKPOINT}"
echo "Clinical-best model: ${CLINICAL_CHECKPOINT}"
echo "BLEU test results: ${EXPORT_BLEU_DIR}/all_test_results.csv"
echo "Clinical test results: ${EXPORT_CLINICAL_DIR}/all_test_results.csv"
echo "============================================================"

