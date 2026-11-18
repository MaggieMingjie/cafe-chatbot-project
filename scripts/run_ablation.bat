@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Ablation Study for mT5-Bilingual Model
echo ========================================

:: Resolve the project root from this script's location.
cd /d "%~dp0.."

:: 设置Python路径，把src加入模块路径
set PYTHONPATH=%cd%
set PYTHON=python
set SCRIPT=src/train_ablation.py

set "BASE_ARGS=--model_name google/mt5-small --language mixed --batch_size 16 --lr 0.0002 --epochs 15"

echo.
echo [1/4] Running Full Model (Baseline)
%PYTHON% %SCRIPT% %BASE_ARGS% --use_lora --joint_loss_lambda 1.2 --confidence_threshold 0.7 --ablation_name full_model
if errorlevel 1 pause

echo.
echo [2/4] Running Variant 1: Without LoRA
%PYTHON% %SCRIPT% %BASE_ARGS% --no_lora --joint_loss_lambda 1.2 --confidence_threshold 0.7 --ablation_name variant1_no_lora
if errorlevel 1 pause

echo.
echo [3/4] Running Variant 2: Without Joint Loss
%PYTHON% %SCRIPT% %BASE_ARGS% --use_lora --joint_loss_lambda 0.0 --confidence_threshold 0.7 --ablation_name variant2_no_joint_loss
if errorlevel 1 pause

echo.
echo [4/4] Running Variant 3: Without Confidence Threshold
%PYTHON% %SCRIPT% %BASE_ARGS% --use_lora --joint_loss_lambda 1.2 --confidence_threshold 0.0 --ablation_name variant3_no_threshold
if errorlevel 1 pause

echo.
echo ======================================================================
echo ✅ All ablation experiments completed successfully!
echo ======================================================================

pause
