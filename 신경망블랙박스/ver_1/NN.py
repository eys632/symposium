# 0. 환경 설정 및 라이브러리 임포트
import os
import tensorflow as tf
import numpy as np

# TensorFlow가 초기화되기 전에 실행하여 2번 GPU만 사용하도록 강제
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f"TensorFlow가 인식하는 GPU: {tf.config.list_physical_devices('GPU')}")


# 1. 데이터 준비 및 전역 변수 설정

# 로그 파일을 저장할 디렉토리 경로
LOG_DIR = "/home/a202192020/내_공부/신경망블랙박스/py파일"
os.makedirs(LOG_DIR, exist_ok=True)

# MNIST 데이터 로드 및 정규화
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 레이블을 이진 형태로 변환 (3이면 1, 아니면 0)
y_train_binary = (y_train == 3).astype(np.float32)
y_test_binary = (y_test == 3).astype(np.float32)


# 방대한 로그 생성을 피하기 위해 데이터의 일부(128개)만 사용하여 실험 진행
# 배치 크기를 64로 설정하면 총 2개의 스텝(Step)이 실행됨
BATCH_SIZE = 64
DATASET_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:DATASET_SIZE], y_train_binary[:DATASET_SIZE])).batch(BATCH_SIZE)


# 2. 실험 실행 함수 정의

def run_experiment(model_type, log_file_name):
    """
    주어진 모델 타입(대칭/표준)에 따라 실험을 실행하고 상세 로그를 기록합니다.
    """
    # 2-1. 모델 생성
    if model_type == 'symmetric':
        # 가설 모델(A): 모든 가중치를 1, 편향을 0으로 초기화
        initializer_weights = tf.keras.initializers.Ones()
        initializer_bias = tf.keras.initializers.Zeros()
        model_name = "모델 A (가설-대칭 모델)"
    else:
        # 표준 모델(B): Keras 기본 초기화 사용 (Glorot Uniform)
        initializer_weights = tf.keras.initializers.GlorotUniform()
        initializer_bias = tf.keras.initializers.Zeros()
        model_name = "모델 B (표준-무작위 모델)"

    # 모델의 각 층 정의
    flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
    hidden_layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer_weights, bias_initializer=initializer_bias)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer_weights, bias_initializer=initializer_bias)
    all_layers = [hidden_layer, output_layer]

    # 2-2. 학습 설정
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    # 2-3. 로그 파일 설정 및 학습 시작
    log_file_path = os.path.join(LOG_DIR, log_file_name)
    
    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"🔬 [{model_name}] 블랙박스 로그 시작\n")
        f.write("="*60 + "\n")

        # 1 에포크만 실행
        for epoch in range(1):
            f.write(f"\n--- EPOCH {epoch+1} ---\n")
            accuracy_metric.reset_states() # 에포크 시작 시 정확도 초기화

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                f.write(f"\n{'='*25} STEP {step+1} {'='*25}\n")

                with tf.GradientTape() as tape:
                    # 3-1. 순전파 (Forward Propagation) 기록
                    f.write("\n--[ 3-1. 순전파 (Forward Propagation) ]--\n")
                    
                    x_flattened = flatten_layer(x_batch)
                    hidden_activations = hidden_layer(x_flattened)
                    
                    f.write("\n[은닉층 정보]\n")
                    f.write(f"  - 가중치 (Shape: {hidden_layer.kernel.shape}):\n{np.array2string(hidden_layer.kernel.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    f.write(f"  - 편향 (Shape: {hidden_layer.bias.shape}):\n{np.array2string(hidden_layer.bias.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    f.write(f"  - 활성화 값 (Shape: {hidden_activations.shape}):\n{np.array2string(hidden_activations.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")

                    logits = output_layer(hidden_activations)
                    
                    f.write("\n[출력층 정보]\n")
                    f.write(f"  - 가중치 (Shape: {output_layer.kernel.shape}):\n{np.array2string(output_layer.kernel.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    f.write(f"  - 편향 (Shape: {output_layer.bias.shape}):\n{np.array2string(output_layer.bias.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    f.write(f"  - 최종 예측 값 (Logits):\n{np.array2string(logits.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    
                    y_batch_expanded = tf.expand_dims(y_batch, axis=1)
                    loss_value = loss_fn(y_batch_expanded, logits)
                    
                    # 3-2. 성능 지표 기록
                    accuracy_metric.update_state(y_batch_expanded, logits)
                    f.write("\n--[ 3-2. 성능 지표 ]--\n")
                    f.write(f"  - 손실 (Loss): {loss_value.numpy()}\n")
                    f.write(f"  - 현재 스텝까지의 누적 정확도 (Cumulative Accuracy): {accuracy_metric.result().numpy()}\n")

                    # 3-3. 역전파 (Backpropagation) 기록
                    f.write("\n--[ 3-3. 역전파 (Backpropagation) ]--\n")
                    trainable_vars = [var for layer in all_layers for var in layer.trainable_variables]
                    grads = tape.gradient(loss_value, trainable_vars)
                
                    for i, layer in enumerate(all_layers):
                        layer_name = '은닉층' if i == 0 else '출력층'
                        f.write(f"\n[{layer_name}의 기울기(Gradients)]\n")
                        f.write(f"  - 가중치 기울기 (Shape: {grads[i*2].shape}):\n{np.array2string(grads[i*2].numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                        f.write(f"  - 편향 기울기 (Shape: {grads[i*2+1].shape}):\n{np.array2string(grads[i*2+1].numpy(), threshold=np.inf, max_line_width=np.inf)}\n")

                    # 3-4. 파라미터 업데이트 기록
                    optimizer.apply_gradients(zip(grads, trainable_vars))
                    
                    f.write("\n--[ 3-4. 파라미터 업데이트 후 ]--\n")
                    for i, layer in enumerate(all_layers):
                        layer_name = '은닉층' if i == 0 else '출력층'
                        f.write(f"\n[{layer_name}의 새 파라미터]\n")
                        f.write(f"  - 새 가중치:\n{np.array2string(layer.kernel.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                        f.write(f"  - 새 편향:\n{np.array2string(layer.bias.numpy(), threshold=np.inf, max_line_width=np.inf)}\n")
                    
                f.write(f"\n{'='*25} STEP {step+1} 종료 {'='*25}\n")
            
            f.write(f"\n\n--- 최종 에포크 정확도: {accuracy_metric.result().numpy()} ---\n")

    print(f"[{model_name}] 실험 완료. 로그가 다음 파일에 저장되었습니다:\n{log_file_path}\n")


# 3. 메인 실행 로직
if __name__ == "__main__":
    # 실험 A: 가설(대칭) 모델 실행
    run_experiment(model_type='symmetric', log_file_name='hypothesis_model_log.txt')
    
    # 실험 B: 표준(무작위) 모델 실행
    run_experiment(model_type='standard', log_file_name='standard_model_log.txt')

    print("="*60)
    print("모든 실험이 성공적으로 완료되었습니다.")
    print("="*60)