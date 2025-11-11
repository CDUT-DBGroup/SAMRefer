#!/bin/bash

# 多数据集训练脚本
# 使用改进的模型架构、训练策略和数据增强，支持多个数据集联合训练

echo "Starting multi-dataset ReferSAM training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 激活conda环境
# source ~/anaconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ovseg_lower_pytorch
conda activate SAM

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=1
export FILELOCK_DEFAULT_CLASS=SoftFileLock

# 启动多数据集训练
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

# 记录日志文件名
LOG_FILE="train_multi_dataset_$(date +%m%d_%H%M).log"
PID_FILE="/tmp/train_enhanced_multi_dataset.pid"
HEALTH_CHECK_INTERVAL=300  # 健康检查间隔（秒）：5分钟
LOG_UPDATE_TIMEOUT=300     # 日志无更新超时（秒）：5分钟
MAX_TRAINING_TIME=432000    # 最大训练时间（秒）：24小时

# 清理函数：杀死所有相关进程
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "train_enhanced_multi_dataset.py" 2>/dev/null
    pkill -f "deepspeed" 2>/dev/null
    rm -f "$PID_FILE"
}

# 设置 trap：在脚本退出时（正常退出、异常退出、被信号中断）自动关机
# 注意：SIGKILL (kill -9) 无法被捕获，但正常情况不会使用
trap 'EXIT_CODE=$?; cleanup; echo "Script exiting with code $EXIT_CODE. Shutting down..."; /usr/bin/shutdown -h now' EXIT INT TERM

echo "Starting training... Log file: $LOG_FILE"
echo "To monitor training: tail -f $LOG_FILE"
echo "To stop training: pkill -f train_enhanced_multi_dataset.py"

# 健康检查函数：监控训练进程是否卡住
health_check() {
    local log_file="$1"
    local pid_file="$2"
    
    while true; do
        sleep $HEALTH_CHECK_INTERVAL
        
        # 检查进程是否还在运行
        if [ -f "$pid_file" ]; then
            local main_pid=$(cat "$pid_file" 2>/dev/null)
            if [ -z "$main_pid" ] || ! kill -0 "$main_pid" 2>/dev/null; then
                echo "[Health Check] Main process $main_pid is not running. Exiting health check."
                return 0  # 正常退出，进程已结束
            fi
        else
            # PID 文件不存在，可能训练已结束
            echo "[Health Check] PID file not found. Exiting health check."
            return 0
        fi
        
        # 检查日志文件是否有更新（最近 LOG_UPDATE_TIMEOUT 秒内）
        if [ -f "$log_file" ]; then
            local last_modified=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)
            local current_time=$(date +%s)
            local time_diff=$((current_time - last_modified))
            
            if [ $time_diff -gt $LOG_UPDATE_TIMEOUT ]; then
                echo "[Health Check] WARNING: Log file has not been updated for ${time_diff} seconds (> ${LOG_UPDATE_TIMEOUT}s)."
                echo "[Health Check] Training may be stuck. Attempting to kill training processes..."
                
                # 只杀死训练相关进程，不杀死健康检查本身
                pkill -f "train_enhanced_multi_dataset.py" 2>/dev/null
                pkill -f "deepspeed.*train_enhanced_multi_dataset" 2>/dev/null
                
                # 如果主进程还在，尝试杀死它
                if [ -f "$pid_file" ]; then
                    local main_pid=$(cat "$pid_file" 2>/dev/null)
                    if kill -0 "$main_pid" 2>/dev/null; then
                        kill -TERM "$main_pid" 2>/dev/null
                        sleep 3
                        if kill -0 "$main_pid" 2>/dev/null; then
                            kill -KILL "$main_pid" 2>/dev/null
                        fi
                    fi
                fi
                
                return 1  # 返回错误状态
            fi
        fi
    done
}

# 启动训练命令（后台运行，记录PID）
deepspeed --num_gpus $NUM_GPUS train_enhanced_multi_dataset.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/main_refersam_bert.yaml \
    --use_enhanced_loss \
    --loss_config_path configs/enhanced_loss_config.yaml \
     > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"
echo "Training started with PID: $TRAIN_PID"

# 启动健康检查（后台运行）
health_check "$LOG_FILE" "$PID_FILE" &
HEALTH_CHECK_PID=$!

# 等待训练进程结束，使用超时机制
# 如果训练超过 MAX_TRAINING_TIME 秒，自动终止
TRAIN_EXIT_CODE=0
HEALTH_CHECK_FAILED=0
elapsed_time=0

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 10
    elapsed_time=$((elapsed_time + 10))
    
    # 检查健康检查进程是否还在运行
    if ! kill -0 $HEALTH_CHECK_PID 2>/dev/null; then
        # 健康检查进程已退出，检查退出状态
        wait $HEALTH_CHECK_PID 2>/dev/null
        HEALTH_CHECK_EXIT=$?
        if [ $HEALTH_CHECK_EXIT -ne 0 ]; then
            echo "Health check detected training stuck. Process may have been killed."
            HEALTH_CHECK_FAILED=1
            # 等待一下，看训练进程是否真的被杀死
            sleep 5
            if ! kill -0 $TRAIN_PID 2>/dev/null; then
                break
            fi
        fi
    fi
    
    # 检查是否超过最大训练时间
    if [ $elapsed_time -ge $MAX_TRAINING_TIME ]; then
        echo "Training exceeded maximum time limit (${MAX_TRAINING_TIME}s). Killing process..."
        kill -TERM $TRAIN_PID 2>/dev/null
        sleep 5
        # 如果进程还在运行，强制杀死
        if kill -0 $TRAIN_PID 2>/dev/null; then
            kill -KILL $TRAIN_PID 2>/dev/null
        fi
        TRAIN_EXIT_CODE=124
        break
    fi
done

# 等待进程完全结束并获取退出状态码
wait $TRAIN_PID 2>/dev/null
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    TRAIN_EXIT_CODE=$?
fi

# 如果健康检查检测到卡住，设置退出码
if [ $HEALTH_CHECK_FAILED -eq 1 ] && [ $TRAIN_EXIT_CODE -eq 0 ]; then
    TRAIN_EXIT_CODE=125  # 自定义退出码：健康检查检测到卡住
fi

# 停止健康检查（如果还在运行）
kill $HEALTH_CHECK_PID 2>/dev/null
wait $HEALTH_CHECK_PID 2>/dev/null

# 清理 PID 文件
rm -f "$PID_FILE"

# 根据退出状态码判断训练结果
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
    deepspeed --num_gpus $NUM_GPUS validate_bert.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/main_refersam_bert.yaml \
    --use_enhanced_loss \
    --loss_config_path configs/enhanced_loss_config.yaml \
     > validate_$(date +%m%d_%H%M).log 2>&1 &
    VALIDATE_PID=$!
    echo $VALIDATE_PID > "$VALIDATE_PID_FILE"
    echo "Validation started with PID: $VALIDATE_PID"
elif [ $TRAIN_EXIT_CODE -eq 124 ]; then
    echo "Training exceeded maximum time limit (${MAX_TRAINING_TIME}s)."
    cleanup
elif [ $TRAIN_EXIT_CODE -eq 125 ]; then
    echo "Training detected as stuck by health check (log file not updated for > ${LOG_UPDATE_TIMEOUT}s)."
    cleanup
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE."
    cleanup
fi

# 脚本正常结束，trap 会自动执行关机命令
