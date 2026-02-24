import numpy as np
import sympy as sp
import math
import scipy.special
from decimal import Decimal
import scipy.integrate as spi
import mpmath
from sympy.core.evalf import evalf
import cmath
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from pandas import DataFrame
import pandas as pd

pmax = 0.8
# fmax = 2e9
fmc = 10e9
T = 0.05
N = 80
wmax1 = 3
wmax2 = 3
reta1 = 1000000
reta2 = 1000000
la = 0.1
bao = 100000
as1 = 100
ac1 = 100
as2 = 100
ac2 = 100
bs1 = 1e6
bc1 = 1e6
bs2 = 1e6
bc2 = 1e6
ms1 = 2
ms2 = 2
mc1 = 1
mc2 = 1
thet = 0.00003
o = 1
e = 1e-7
sigmas = 0.000000000001
sigmac = 0.000000000001
amc = 500
v1 = v2 = v3 = v4 = v5 = v6 = v9 = 0.001
v7 = v8 = 0.01

t_ex_points = []
t_ex_points1 = []
points = []
t_points = []
t_points1 = []
t_1_points = []
t_1_points1 = []
t_2_points = []
t_2_points1 = []
t_re_points = []
t_re_points1 = []

L_points = []
L_points1 = []
L_1_points = []
L_2_points = []
L_1_points1 = []
L_2_points1 = []
L_re_points = []
L_re_points1 = []

# 生成随机信道系数 (0到1之间)
num_samples = 1000
num_features = 100
random.seed(2)
X = np.random.rand(num_samples, num_features).astype(np.float32)
# 定义DNN模型参数
input_dim = num_features
hidden_dim1 = 64
hidden_dim2 = 32
hidden_dim3 = 16
output_dim = 7  # 功率、计算频率、卸载策略
res_dim1 = 1
res_dim2 = 64
res_dim3 = 32
resout_dim = 1

# 定义DNN模型参数1
input_dim_1 = num_features
hidden_dim1_1 = 64
hidden_dim2_1 = 32
hidden_dim3_1 = 16
output_dim_1 = 5  # 功率、计算频率、卸载策略
# 定义DNN模型参数2
input_dim_2 = num_features
hidden_dim1_2 = 64
hidden_dim2_2 = 32
hidden_dim3_2 = 16
output_dim_2 = 5  # 功率、计算频率、卸载策略
l2 = 0.001

# 初始化权重和偏置

def initialize_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1, seed=32), trainable=True)


def initialize_weights1(shape):
    initializer = tf.keras.initializers.HeNormal()
    return tf.Variable(initializer(shape))

# 无监督学习
W1 = initialize_weights([input_dim, hidden_dim1])
b1 = tf.Variable(tf.zeros([hidden_dim1]), trainable=True)
W2 = initialize_weights([hidden_dim1, hidden_dim2])
b2 = tf.Variable(tf.zeros([hidden_dim2]), trainable=True)
W3 = initialize_weights([hidden_dim2, hidden_dim3])
b3 = tf.Variable(tf.zeros([hidden_dim3]), trainable=True)
W4 = initialize_weights([hidden_dim3, output_dim])
b4 = tf.Variable(tf.zeros([output_dim]), trainable=True)
# 正则化
W1_re = initialize_weights([input_dim, hidden_dim1])
b1_re = tf.Variable(tf.zeros([hidden_dim1]), trainable=True)
W2_re = initialize_weights([hidden_dim1, hidden_dim2])
b2_re = tf.Variable(tf.zeros([hidden_dim2]), trainable=True)
W3_re = initialize_weights([hidden_dim2, hidden_dim3])
b3_re = tf.Variable(tf.zeros([hidden_dim3]), trainable=True)
W4_re = initialize_weights([hidden_dim3, output_dim])
b4_re = tf.Variable(tf.zeros([output_dim]), trainable=True)

# 一般值分解（VDN)
W11 = initialize_weights([input_dim_1, hidden_dim1_1])
b11 = tf.Variable(tf.zeros([hidden_dim1_1]), trainable=True)
W21 = initialize_weights([hidden_dim1_1, hidden_dim2_1])
b21 = tf.Variable(tf.zeros([hidden_dim2_1]), trainable=True)
W31 = initialize_weights([hidden_dim2_1, hidden_dim3_1])
b31 = tf.Variable(tf.zeros([hidden_dim3_1]), trainable=True)
W41 = initialize_weights([hidden_dim3_1, output_dim_1])
b41 = tf.Variable(tf.zeros([output_dim_1]), trainable=True)

W12 = initialize_weights([input_dim_2, hidden_dim1_2])
b12 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W22 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b22 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W32 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b32 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W42 = initialize_weights([hidden_dim3_2, output_dim_2])
b42 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

W11_2 = initialize_weights([input_dim_1, hidden_dim1_1])
b11_2 = tf.Variable(tf.zeros([hidden_dim1_1]), trainable=True)
W21_2 = initialize_weights([hidden_dim1_1, hidden_dim2_1])
b21_2 = tf.Variable(tf.zeros([hidden_dim2_1]), trainable=True)
W31_2 = initialize_weights([hidden_dim2_1, hidden_dim3_1])
b31_2 = tf.Variable(tf.zeros([hidden_dim3_1]), trainable=True)
W41_2 = initialize_weights([hidden_dim3_1, output_dim_1])
b41_2 = tf.Variable(tf.zeros([output_dim_1]), trainable=True)

W12_2 = initialize_weights([input_dim_2, hidden_dim1_2])
b12_2 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W22_2 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b22_2 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W32_2 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b32_2 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W42_2 = initialize_weights([hidden_dim3_2, output_dim_2])
b42_2 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

# 加残差网络的分解（RDN）
res_w1 = initialize_weights1([res_dim1, res_dim2])
res_b1 = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2 = initialize_weights1([res_dim2, res_dim3])
res_b2 = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3 = initialize_weights1([res_dim3, resout_dim])
res_b3 = tf.Variable(tf.zeros([resout_dim]), trainable=True)

res_w1_2 = initialize_weights1([res_dim1, res_dim2])
res_b1_2 = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_2 = initialize_weights1([res_dim2, res_dim3])
res_b2_2 = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_2 = initialize_weights1([res_dim3, resout_dim])
res_b3_2 = tf.Variable(tf.zeros([resout_dim]), trainable=True)

res_w1_total = initialize_weights1([res_dim1, res_dim2])
res_b1_total = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_total = initialize_weights1([res_dim2, res_dim3])
res_b2_total = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_total = initialize_weights1([res_dim3, resout_dim])
res_b3_total = tf.Variable(tf.zeros([resout_dim]), trainable=True)

# 超参数
learning_rate = 0.001
epochs = 100
batch_size = 300

# 定义拉格朗日乘子
lambda1 = tf.Variable(0.3, trainable=False)
lambda2 = tf.Variable(0.3, trainable=False)
lambda3 = tf.Variable(0.3, trainable=False)
lambda4 = tf.Variable(0.3, trainable=False)
lambda5 = tf.Variable(0.3, trainable=False)
lambda1_re = tf.Variable(0.3, trainable=False)
lambda2_re = tf.Variable(0.3, trainable=False)
lambda3_re = tf.Variable(0.3, trainable=False)
lambda4_re = tf.Variable(0.3, trainable=False)
lambda5_re = tf.Variable(0.3, trainable=False)

lambda1_vdn1 = tf.Variable(0.3, trainable=False)
lambda2_vdn1 = tf.Variable(0.3, trainable=False)
lambda3_vdn1 = tf.Variable(0.3, trainable=False)

lambda1_vdn2 = tf.Variable(0.3, trainable=False)
lambda2_vdn2 = tf.Variable(0.3, trainable=False)
lambda3_vdn2 = tf.Variable(0.3, trainable=False)

lambda1_vdn1_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn1_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn1_2 = tf.Variable(0.3, trainable=False)

lambda1_vdn2_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn2_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn2_2 = tf.Variable(0.3, trainable=False)

# 使用Adam优化器
optimizer = tf.optimizers.Adam(learning_rate)
optimizer1 = tf.optimizers.Adam(learning_rate)
optimizer11 = tf.optimizers.Adam(learning_rate)
optimizer2 = tf.optimizers.Adam(learning_rate)
optimizer3 = tf.optimizers.Adam(learning_rate)
optimizer4 = tf.optimizers.Adam(learning_rate)
optimizer5 = tf.optimizers.Adam(learning_rate)
optimizer6 = tf.optimizers.Adam(learning_rate)
optimizer7 = tf.optimizers.Adam(learning_rate)


def delay(x, y, beta, z):
    # local computing
    rAk = la * (math.exp(thet * bao) - 1)/thet
    kummer = mpmath.hyperu(ms1, ms1 + 1 - thet * bs1 * T / np.log(2),
                            ms1 / (float(x) * float(pmax) / sigmas))
    kummer = float(kummer)
    log_U1s = np.log((ms1 / (x * pmax / sigmas)) ** ms1) + np.log(kummer)
    U1s = np.exp(log_U1s)
    rSk1 = -1 / thet * np.log(U1s)
    rSk_1 = min(rSk1, T * y * fmax / as1, T * (1 - y) * fmax / ac1) - o
    D = -(np.log(1 - np.exp(-thet * (rSk1 - rSk_1))) + np.log(
        1 - math.exp(-thet * (T * y * fmax / as1 - rSk_1))) + np.log(
        1 - math.exp(-thet * (T * (1 - y) * fmax / ac1 - rSk_1))) + np.log(thet * e * (rSk_1 - rAk)))
    wk1 = 1 / (thet * rSk_1) * D + 1
    # egde computing
    kummerc = mpmath.hyperu(mc1, mc1 + 1 - thet * bc1 * T / np.log(2),
                             mc1 / ((1 - float(x)) * float(pmax) / sigmac))
    log_U1c = np.log((mc1 / ((1 - x) * pmax / sigmac)) ** mc1) + np.log(float(kummerc))
    U1c = math.exp(log_U1c)
    rS3 = -1 / thet * np.log(U1c)
    rS4 = z * T * fmc / amc
    rS_1 = min(rSk1, T * y * fmax / as1, rS3, rS4) - o
    D_1 = -(np.log(1 - math.exp(-thet * (rSk1 - rS_1))) + np.log(
        1 - math.exp(-thet * (T * y * fmax / as1 - rS_1))) + np.log(
        1 - math.exp(-thet * (rS3 - rS_1))) + np.log(1 - math.exp(-thet * (rS4 - rS_1))) + np.log(
        thet * e * (rS_1 - rAk)))
    w1 = 1 / (thet * rSk_1) * D_1 + 1
    t = beta * w1 + (1-beta) * wk1
   #edge computing

    return t, w1, wk1

# x_ex = round(random.uniform(0.2, 0.9), 22)
# y_ex = round(random.uniform(0.2, 0.6), 22)
# beta1_ex = round(random.uniform(0, 1), 22)
# beta2_ex = round(random.uniform(0, 1), 22)
# z_ex = round(random.uniform(0.2, 0.9), 22)
# w1,g1,hg2 = delay(x_ex, y_ex, beta1_ex, z_ex)
# w2,gg1,gg2 = delay(x_ex, y_ex, beta2_ex, 1-z_ex)
# t_ex = 1/2*(w1+w2)
for fmax in np.arange(3,7.5,0.5):
    fmax = fmax*1e9
    for epoch in range(epochs):
        num_batches = 1
        # num_batches = num_samples // batch_size
        for i in range(num_batches):
            batch_X = X[i * batch_size: (i + 1) * batch_size]
            with tf.GradientTape() as tape:
                # 前向传播
                h1 = tf.nn.relu(tf.matmul(batch_X, W1) + b1)
                h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
                h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
                output = tf.nn.sigmoid(tf.matmul(h3, W4) + b4)  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11 = output[:, 0]
                p12 = 0.1 + 0.2 * output[:, 1]
                p21 = output[:, 2]
                p22 = 0.2 + 0.2 * output[:, 3]
                p31 = output[:, 4]
                p32 = output[:, 5]
                p41 = output[:, 6]

                x1 = tf.reduce_mean(p11)
                x2 = tf.reduce_mean(p12)
                y1 = tf.reduce_mean(p21)
                y2 = tf.reduce_mean(p22)
                beta1 = tf.reduce_mean(p31)
                beta2 = tf.reduce_mean(p32)
                z = tf.reduce_mean(p41)
                # user1
                t1,w1,wk1 = delay(x1,y1,beta1,z)
                # user2
                t2,w2,wk2 = delay(x2, y2, beta2, 1-z)
                # E2E delay
                w = 1 / 2 * (t1 + t2)
                MI1 = 0.05 * N * bs1 * np.log2(1 + x1 * pmax / sigmas)
                MI2 = 0.05 * N * bs1 * np.log2(1 + x2 * pmax / sigmas)
                pmax = tf.constant(pmax, dtype=tf.float32)
                L = w + (lambda1 * (reta1 - MI1) + lambda2 * (reta2 - MI2)) + (
                        lambda3 * (t1 - wmax1) + lambda4 * (t2 - wmax2))

            with tf.GradientTape() as tape11:
                # 前向传播
                h1_re = tf.nn.relu(tf.matmul(batch_X, W1_re) + b1_re)
                h2_re = tf.nn.relu(tf.matmul(h1_re, W2_re) + b2_re)
                h3_re = tf.nn.relu(tf.matmul(h2_re, W3_re) + b3_re)
                output_re = tf.nn.sigmoid(tf.matmul(h3_re, W4_re) + b4_re)  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11_re = output_re[:, 0]
                p12_re = 0.2 + 0.2 * output_re[:, 1]
                p21_re = output_re[:, 2]
                p22_re = 0.2 + 0.2 * output_re[:, 3]
                p31_re = output_re[:, 4]
                p32_re = output_re[:, 5]
                p41_re = output_re[:, 6]

                x1_re = tf.reduce_mean(p11_re)
                x2_re = tf.reduce_mean(p12_re)
                y1_re = tf.reduce_mean(p21_re)
                y2_re = tf.reduce_mean(p22_re)
                beta1_re = tf.reduce_mean(p31_re)
                beta2_re = tf.reduce_mean(p32_re)
                z_re = tf.reduce_mean(p41_re)
                # user1
                t1_re, w1_re, wk1_re = delay(x1_re, y1_re, beta1_re, z_re)
                t2_re, w2_re, wk2_re = delay(x2_re, y2_re, beta2_re, 1-z_re)
                w_re = 1 / 2 * (t1_re + t2_re)
                pmax = tf.constant(pmax, dtype=tf.float32)
                # CRB
                # CRB1 = 0.01/x1/pmax
                # CRB2 = 0.01 / x2 / pmax
                MI1_re = 0.05 * bs1 * np.log2(1 + x1_re * pmax / sigmas)
                MI2_re = 0.05 * bs1 * np.log2(1 + x2_re * pmax / sigmas)
                # CRB1_re = (ga + gb) / (x1_re * pmax * (ga * gb - gc * gc))
                # CRB2_re = (ga + gb) / (x2_re * pmax * (ga * gb - gc * gc))
                tt = math.pow(lambda1_re * (MI1_re - reta1), 2) + math.pow(lambda2_re * (MI2_re - reta2), 2) + math.pow(
                    lambda3_re * (t1_re - wmax1), 2)
                L_re = w_re + (lambda1_re * (reta1 - MI1_re) + lambda2_re * (reta2 - MI2_re)) + (
                        lambda3_re * (t1_re - wmax1) + lambda4_re * (t2_re - wmax2)) + 0.5 * tt

            with tf.GradientTape() as tape1:
                # 前向传播
                h1_vdn1 = tf.nn.relu(tf.matmul(batch_X, W11) + b11)
                h2_vdn1 = tf.nn.relu(tf.matmul(h1_vdn1, W21) + b21)
                h3_vdn1 = tf.nn.relu(tf.matmul(h2_vdn1, W31) + b31)
                output_vdn1 = tf.nn.sigmoid(tf.matmul(h3_vdn1, W41 + b41))  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11_vdn1 = 0.2 + 0.2 * output_vdn1[:, 0]
                p21_vdn1 = 0.2 + 0.2 * output_vdn1[:, 1]
                p31_vdn1 = output_vdn1[:, 2]
                p41_vdn1 = output_vdn1[:, 3]
                p51_vdn1 = output_vdn1[:, 4]
                p_vdn1 = tf.reduce_mean(p11_vdn1)
                f_vdn1 = tf.reduce_mean(p21_vdn1)
                beta1_vdn1 = tf.reduce_mean(p31_vdn1)
                z_vdn1 = tf.reduce_mean(p51_vdn1)

                t_vdn1,w_vdn1,wk_vdn1 = delay(p_vdn1,f_vdn1,beta1_vdn1,z_vdn1)
                pmax = tf.constant(pmax, dtype=tf.float32)
                # CRB
                # A_vdn1 = (ga + gb) / (p_vdn1 * pmax * (ga * gb - gc * gc))
                A_vdn1 = 0.06 *  bs1 * np.log2(1 + p_vdn1 * pmax / sigmas)
                L_vdn1 = wk_vdn1 + lambda1_vdn1 * (reta1 - A_vdn1)

            with tf.GradientTape() as tape2:
                # 前向传播
                h1_vdn2 = tf.nn.relu(tf.matmul(batch_X, W12) + b12)
                h2_vdn2 = tf.nn.relu(tf.matmul(h1_vdn2, W22) + b22)
                h3_vdn2 = tf.nn.relu(tf.matmul(h2_vdn2, W32) + b32)
                output_vdn2 = tf.nn.sigmoid(tf.matmul(h3_vdn2, W42) + b42)  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11_vdn2 = output_vdn2[:, 0]
                p21_vdn2 = output_vdn2[:, 1]
                p31_vdn2 = output_vdn2[:, 2]
                p41_vdn2 = output_vdn2[:, 3]
                p51_vdn2 = output_vdn2[:, 4]
                p_vdn2 = tf.reduce_mean(p11_vdn2)
                f_vdn2 = tf.reduce_mean(p21_vdn2)
                beta1_vdn2 = tf.reduce_mean(p31_vdn2)
                beta2_vdn2 = tf.reduce_mean(p41_vdn2)
                z_vdn2 = tf.reduce_mean(p51_vdn2)
                t_vdn2, w_vdn2, wk_vdn2 = delay(p_vdn2, f_vdn2, beta1_vdn1, z_vdn1)
                pmax = tf.constant(pmax, dtype=tf.float32)
                # CRB
                # A_vdn2 = (ga + gb) / (p_vdn2 * pmax * (ga * gb - gc * gc))
                A_vdn2 = 0.06 * bs1 * np.log2(1 + p_vdn2 * pmax / sigmas)
                L_vdn2 = w_vdn2 + lambda1_vdn2 * (reta1 - A_vdn2)

            with tf.GradientTape() as tape3:
                # 前向传播
                h1_vdn1_2 = tf.nn.relu(tf.matmul(batch_X, W11_2) + b11_2)
                h2_vdn1_2 = tf.nn.relu(tf.matmul(h1_vdn1_2, W21_2) + b21_2)
                h3_vdn1_2 = tf.nn.relu(tf.matmul(h2_vdn1_2, W31_2) + b31_2)
                output_vdn1_2 = tf.nn.sigmoid(tf.matmul(h3_vdn1_2, W41_2 + b41_2))  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11_vdn1_2 = output_vdn1_2[:, 0]
                p21_vdn1_2 = output_vdn1_2[:, 1]
                p31_vdn1_2 = output_vdn1_2[:, 2]
                p41_vdn1_2 = output_vdn1_2[:, 3]
                p51_vdn1_2 = output_vdn1_2[:, 4]
                p_vdn1_2 = tf.reduce_mean(p11_vdn1_2)
                f_vdn1_2 = tf.reduce_mean(p21_vdn1_2)
                beta1_vdn1_2 = tf.reduce_mean(p31_vdn1_2)
                beta2_vdn1_2 = tf.reduce_mean(p41_vdn1_2)
                z_vdn1_2 = tf.reduce_mean(p51_vdn1_2)
                t_vdn1_2, w_vdn1_2, wk_vdn1_2 = delay(p_vdn1_2, f_vdn1_2, beta2_vdn1_2, 1-z_vdn1)
                pmax = tf.constant(pmax, dtype=tf.float32)
                # CRB
                # A_vdn1_2 = (ga + gb) / (p_vdn1_2 * pmax * (ga * gb - gc * gc))
                A_vdn1_2 = 0.06 *  bs1 * np.log2(1 + p_vdn1_2 * pmax / sigmas)
                L_vdn1_2 = wk_vdn1_2 + lambda1_vdn1_2 * (reta1 - A_vdn1_2) + lambda2_vdn1_2 * (
                            wk_vdn1_2 - wmax1)

            with tf.GradientTape() as tape4:
                # 前向传播
                h1_vdn2_2 = tf.nn.relu(tf.matmul(batch_X, W12_2) + b12_2)
                h2_vdn2_2 = tf.nn.relu(tf.matmul(h1_vdn2_2, W22_2) + b22_2)
                h3_vdn2_2 = tf.nn.relu(tf.matmul(h2_vdn2_2, W32_2) + b32_2)
                output_vdn2_2 = tf.nn.sigmoid(tf.matmul(h3_vdn2_2, W42_2) + b42_2)  # 使用 sigmoid 确保输出在 0 到 1 之间
                # 提取功率、计算频率和卸载策略
                p11_vdn2_2 = 0.2 + 0.2 * output_vdn2_2[:, 0]
                p21_vdn2_2 = 0.2 + 0.2 * output_vdn2_2[:, 1]
                p31_vdn2_2 = output_vdn2_2[:, 2]
                p41_vdn2_2 = output_vdn2_2[:, 3]
                p51_vdn2_2 = output_vdn2_2[:, 4]
                p_vdn2_2 = tf.reduce_mean(p11_vdn2_2)
                f_vdn2_2 = tf.reduce_mean(p21_vdn2_2)
                beta1_vdn2_2 = tf.reduce_mean(p31_vdn2_2)
                beta2_vdn2_2 = tf.reduce_mean(p41_vdn2_2)
                z_vdn2_2 = tf.reduce_mean(p51_vdn2_2)
                t_vdn2_2, w_vdn2_2, wk_vdn2_2 = delay(p_vdn2_2, f_vdn2_2, beta2_vdn1_2, 1-z_vdn1)
                pmax = tf.constant(pmax, dtype=tf.float32)
                # CRB
                # A_vdn2_2 = (ga + gb) / (p_vdn2_2 * pmax * (ga * gb - gc * gc))
                A_vdn2_2 = 0.05 *  bs1 * np.log2(1 + p_vdn2_2 * pmax / sigmas)
                L_vdn2_2 = w_vdn2_2 + lambda1_vdn2_2 * (reta1 - A_vdn2_2) + lambda2_vdn2_2 * (
                            w_vdn2_2 - wmax1)
            # t_c1 = (1 - beta1_vdn1) *  w_vdn1 + beta1_vdn1 * w_vdn2
            # t_c2 = (1 - beta2_vdn2) * w_vdn1 + beta2_vdn2 * w_vdn2
            # t_c3 = (1 - beta1_vdn1_2) *  w_vdn1 + beta1_vdn1_2 * w_vdn2
            # t_c4 = (1 - beta2_vdn2_2) * w_vdn1 + beta2_vdn2_2 * w_vdn2
            # t_de = 1/2 * (1/2 * (t_c1 + t_c2) + 1/2 * (t_c3 + t_c4))
            t_de = 1/2 * ((1 - beta1_vdn1) * w_vdn1 + beta1_vdn1 * w_vdn2 + (1 - beta2_vdn2_2) * w_vdn1_2 + beta2_vdn2_2 * w_vdn2_2)
            # 残差连接
            with tf.GradientTape() as tape5:
                t_res = tf.stack([t_vdn1, t_vdn2])
                t_res = tf.reshape(t_res, [2, 1])
                res1 = tf.nn.relu(tf.matmul(t_res, res_w1) + res_b1)
                res2 = tf.nn.relu(tf.matmul(res1, res_w2) + res_b2)
                res3 = 0.01 * tf.nn.tanh(tf.matmul(res2, res_w3) + res_b3)
                t_with_res = tf.nn.relu(t_res + res3)

            with tf.GradientTape() as tape6:
                t_res_2 = tf.stack([t_vdn1_2, t_vdn2_2])
                t_res_2 = tf.reshape(t_res_2, [2, 1])
                res1_2 = tf.nn.relu(tf.matmul(t_res_2, res_w1_2) + res_b1_2)
                res2_2 = tf.nn.relu(tf.matmul(res1_2, res_w2_2) + res_b2_2)
                res3_2 = 0.01 * tf.nn.tanh(tf.matmul(res2_2, res_w3_2) + res_b3_2)
                t_with_res_2 = tf.nn.relu(t_res_2 + res3_2)

            t_total_rdn1 = (1 - beta1_vdn1) * t_with_res[0][0] + beta1_vdn1 * t_with_res[1][0]
            t_total_rdn2 = (1 - beta1_vdn2) * t_with_res[0][0] + beta1_vdn2 * t_with_res[1][0]
            t_total_rdn_1 = 1 / 2 * (t_total_rdn1 + t_total_rdn2)
            t_total_rdn1_2 = (1 - beta2_vdn1_2) * t_with_res_2[0][0] + beta2_vdn1_2 * t_with_res_2[1][0]
            t_total_rdn2_2 = (1 - beta2_vdn2_2) * t_with_res_2[0][0] + beta2_vdn2_2 * t_with_res_2[1][0]
            t_total_rdn_2 = 1 / 2 * (t_total_rdn1_2 + t_total_rdn2_2)

            with tf.GradientTape() as tape7:
                t_res_total = tf.stack([t_total_rdn1, t_total_rdn1_2])
                t_res_total = tf.reshape(t_res_total, [2, 1])
                res1_total = tf.nn.relu(tf.matmul(t_res_total, res_w1_total) + res_b1_total)
                res2_total = tf.nn.relu(tf.matmul(res1_total, res_w2_total) + res_b2_total)
                res3_total = 0.01 * tf.nn.tanh(tf.matmul(res2_total, res_w3_total) + res_b3_total)
                t_with_res_total = tf.nn.relu(t_res_total + res3_total)

            t_total_rdn = 1 / 2 * (t_with_res_total[0][0] + t_with_res_total[1][0])

            # 参数更新
            gradients = tape.gradient(L, [W1, b1, W2, b2, W3, b3, W4, b4])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3, W4, b4]))
            # 更新拉格朗日乘子
            lambda1 = lambda1 - v5 * (reta1 - MI1)
            lambda2 = lambda2 - v6 * (reta2 - MI2)
            lambda3 = lambda3 - v7 * (w1 - wmax1)
            lambda4 = lambda4 - v8 * (w2 - wmax2)

            gradients11 = tape11.gradient(L_re, [W1_re, b1_re, W2_re, b2_re, W3_re, b3_re, W4_re, b4_re])
            optimizer11.apply_gradients(zip(gradients11, [W1_re, b1_re, W2_re, b2_re, W3_re, b3_re, W4_re, b4_re]))
            # 更新拉格朗日乘子
            lambda1_re = lambda1_re - v5 * (reta1 - MI1_re)
            lambda2_re = lambda2_re - v6 * (reta2 - MI2_re)
            lambda3_re = lambda3_re - v7 * (w1_re - wmax1)
            lambda4_re = lambda4_re - v8 * (w2_re - wmax2)

            gradients_1 = tape1.gradient(L_vdn1, [W11, b11, W21, b21, W31, b31, W41, b41])
            optimizer1.apply_gradients(zip(gradients_1, [W11, b11, W21, b21, W31, b31, W41, b41]))
            lambda1_vdn1 = lambda1_vdn1 - v5 * (reta1 - A_vdn1)
            lambda2_vdn1 = lambda2_vdn1 - v7 * (wk_vdn1 - wmax1)

            gradients_2 = tape2.gradient(L_vdn2, [W12, b12, W22, b22, W32, b32, W42, b42])
            optimizer2.apply_gradients(zip(gradients_2, [W12, b12, W22, b22, W32, b32, W42, b42]))
            lambda1_vdn2 = lambda1_vdn2 - v5 * (reta1 - A_vdn2)
            lambda2_vdn2 = lambda2_vdn2 - v7 * (wk_vdn2 - wmax1)

            gradients_3 = tape3.gradient(L_vdn1_2, [W11_2, b11_2, W21_2, b21_2, W31_2, b31_2, W41_2, b41_2])
            optimizer3.apply_gradients(zip(gradients_3, [W11_2, b11_2, W21_2, b21_2, W31_2, b31_2, W41_2, b41_2]))
            lambda1_vdn1_2 = lambda1_vdn1_2 - v5 * (reta1 - A_vdn1_2)
            lambda2_vdn1_2 = lambda2_vdn1_2 - v7 * (wk_vdn1_2 - wmax1)

            gradients_4 = tape4.gradient(L_vdn2_2, [W12_2, b12_2, W22_2, b22_2, W32_2, b32_2, W42_2, b42_2])
            optimizer4.apply_gradients(zip(gradients_4, [W12_2, b12_2, W22_2, b22_2, W32_2, b32_2, W42_2, b42_2]))
            lambda1_vdn2_2 = lambda1_vdn2_2 - v5 * (reta1 - A_vdn2_2)
            lambda2_vdn2_2 = lambda2_vdn2_2 - v7 * (wk_vdn2_2 - wmax1)

            gradients_5 = tape5.gradient(t_with_res,
                                         [res_w1, res_b1, res_w2, res_b2, res_w3, res_b3])
            optimizer5.apply_gradients(zip(gradients_5,
                                           [res_w1, res_b1, res_w2, res_b2, res_w3, res_b3]))

            gradients_6 = tape6.gradient(t_with_res_2,
                                         [res_w1_2, res_b1_2, res_w2_2, res_b2_2, res_w3_2, res_b3_2])
            optimizer6.apply_gradients(zip(gradients_6,
                                           [res_w1_2, res_b1_2, res_w2_2, res_b2_2, res_w3_2, res_b3_2]))
            gradients_7 = tape7.gradient(t_with_res_total,
                                         [res_w1_total, res_b1_total, res_w2_total, res_b2_total, res_w3_total,
                                          res_b3_total])
            optimizer7.apply_gradients(zip(gradients_7,
                                           [res_w1_total, res_b1_total, res_w2_total, res_b2_total, res_w3_total,
                                            res_b3_total]))
        print(
            f"Epoch {epoch + 1}")

    points.append(float(fmax/1e9))
    t_points.append(float(w))
    t_re_points.append(float(w_re))
    # t_re_points1.append(w_re)
    # w_de = 1 / 2 * ((1 - beta1_vdn1) * wk_vdn1 + beta1_vdn1 * w_vdn2 + (
    #         1 - beta2_vdn1_2) * wk_vdn1_2 + beta2_vdn1_2 * w_vdn2_2)
    t_1_points.append(float(t_de))
    t_2_points.append(float(t_total_rdn))


# print("4", float(1 /2 * (p_vdn1 + p_vdn2 )), float(1/2 * (p_vdn1_2 + p_vdn2_2)), float(1 / 2 * (f_vdn1 + f_vdn2)), float(1 / 2 * (f_vdn1_2 + f_vdn2_2)),
#       float(beta1_vdn1), float(beta2_vdn1_2), float(z_vdn1))
# print("3", float(x1_re), float(x2_re), float(y1_re), float(y2_re), float(beta1_re), float(beta2_re), float(z_re))
# print("2", float(x1), float(x2), float(y1), float(y2), float(beta1), float(beta2), float(z))
# print("1", float(x_ex), float(y_ex), float(beta1_ex), float(beta2_ex), float(z_ex))

# plt.xlabel('Iterations')
# plt.ylabel('Lagrangian Function')
# plt.legend()
df1 = DataFrame({'number': points, 'value2': t_points, 'value3': t_re_points, 'value4': t_1_points, 'value5': t_2_points})
df1.to_excel('computation_E2Edelay_0225.xlsx',  index=False)
plt.figure()
# plt.plot(points, t_ex_points, color='black', label='exhausting search')
plt.plot(points, t_points, color='red', label='unsupervised learning ')
plt.plot(points, t_re_points, color='blue', label='Learning_regularization')
plt.plot(points, t_1_points, color='purple', label='Learning_distributed')
plt.plot(points, t_2_points, color='green', label='Learning_residual')
plt.xlabel('Iterations')
plt.ylabel('delay')
plt.legend()

plt.show()