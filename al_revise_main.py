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
fmax = 2e9
fmc = 10e9
T = 0.05
N = 50
wmax1 = 3
wmax2 = 3
wmax3 = 3
wmax4 = 3
wmax5 = 3
reta1 = 1000000
reta2 = 1000000
reta3 = 1000000
reta4 = 1000000
reta5 = 1000000
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
bs3 = 1e6
bc3 = 1e6
bs4 = 1e6
bc4 = 1e6
bs5 = 1e6
bc5 = 1e6
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

num_samples = 1000
num_features = 100
random.seed(2)
X = np.random.rand(num_samples, num_features).astype(np.float32)
# 定义DNN模型参数
input_dim = num_features
hidden_dim1 = 64
hidden_dim2 = 32
hidden_dim3 = 16
output_dim = 20 
res_dim1 = 1
res_dim2 = 64
res_dim3 = 32
resout_dim = 1

# 定义DNN模型参数1
input_dim_1 = num_features
hidden_dim1_1 = 64
hidden_dim2_1 = 32
hidden_dim3_1 = 20
output_dim_1 = 5  
# 定义DNN模型参数2
input_dim_2 = num_features
hidden_dim1_2 = 64
hidden_dim2_2 = 32
hidden_dim3_2 = 16
output_dim_2 = 5  
l2 = 0.001

# 初始化权重和偏置

def initialize_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1, seed=42), trainable=True)


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
W5 = initialize_weights([hidden_dim3, output_dim])
b5 = tf.Variable(tf.zeros([output_dim]), trainable=True)
# 正则化
W1_re = initialize_weights([input_dim, hidden_dim1])
b1_re = tf.Variable(tf.zeros([hidden_dim1]), trainable=True)
W2_re = initialize_weights([hidden_dim1, hidden_dim2])
b2_re = tf.Variable(tf.zeros([hidden_dim2]), trainable=True)
W3_re = initialize_weights([hidden_dim2, hidden_dim3])
b3_re = tf.Variable(tf.zeros([hidden_dim3]), trainable=True)
W4_re = initialize_weights([hidden_dim3, output_dim])
b4_re = tf.Variable(tf.zeros([output_dim]), trainable=True)
W5_re = initialize_weights([hidden_dim3, output_dim])
b5_re = tf.Variable(tf.zeros([output_dim]), trainable=True)

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

W13 = initialize_weights([input_dim_2, hidden_dim1_2])
b13 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W23 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b23 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W33 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b33 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W43 = initialize_weights([hidden_dim3_2, output_dim_2])
b43 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

W14 = initialize_weights([input_dim_2, hidden_dim1_2])
b14 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W24 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b24 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W34 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b34 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W44 = initialize_weights([hidden_dim3_2, output_dim_2])
b44 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

W15 = initialize_weights([input_dim_2, hidden_dim1_2])
b15 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W25 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b25 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W35 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b35 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W45 = initialize_weights([hidden_dim3_2, output_dim_2])
b45 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

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

W13_2 = initialize_weights([input_dim_2, hidden_dim1_2])
b13_2 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W23_2 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b23_2 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W33_2 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b33_2 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W43_2 = initialize_weights([hidden_dim3_2, output_dim_2])
b43_2 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

W14_2 = initialize_weights([input_dim_2, hidden_dim1_2])
b14_2 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W24_2 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b24_2 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W34_2 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b34_2 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W44_2 = initialize_weights([hidden_dim3_2, output_dim_2])
b44_2 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

W15_2 = initialize_weights([input_dim_2, hidden_dim1_2])
b15_2 = tf.Variable(tf.zeros([hidden_dim1_2]), trainable=True)
W25_2 = initialize_weights([hidden_dim1_2, hidden_dim2_2])
b25_2 = tf.Variable(tf.zeros([hidden_dim2_2]), trainable=True)
W35_2 = initialize_weights([hidden_dim2_2, hidden_dim3_2])
b35_2 = tf.Variable(tf.zeros([hidden_dim3_2]), trainable=True)
W45_2 = initialize_weights([hidden_dim3_2, output_dim_2])
b45_2 = tf.Variable(tf.zeros([output_dim_2]), trainable=True)

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

res_w1_3 = initialize_weights1([res_dim1, res_dim2])
res_b1_3 = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_3 = initialize_weights1([res_dim2, res_dim3])
res_b2_3 = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_3 = initialize_weights1([res_dim3, resout_dim])
res_b3_3 = tf.Variable(tf.zeros([resout_dim]), trainable=True)

res_w1_4 = initialize_weights1([res_dim1, res_dim2])
res_b1_4 = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_4 = initialize_weights1([res_dim2, res_dim3])
res_b2_4 = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_4 = initialize_weights1([res_dim3, resout_dim])
res_b3_4 = tf.Variable(tf.zeros([resout_dim]), trainable=True)

res_w1_5 = initialize_weights1([res_dim1, res_dim2])
res_b1_5 = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_5 = initialize_weights1([res_dim2, res_dim3])
res_b2_5 = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_5 = initialize_weights1([res_dim3, resout_dim])
res_b3_5 = tf.Variable(tf.zeros([resout_dim]), trainable=True)

res_w1_total = initialize_weights1([res_dim1, res_dim2])
res_b1_total = tf.Variable(tf.zeros([res_dim2]), trainable=True)
res_w2_total = initialize_weights1([res_dim2, res_dim3])
res_b2_total = tf.Variable(tf.zeros([res_dim3]), trainable=True)
res_w3_total = initialize_weights1([res_dim3, resout_dim])
res_b3_total = tf.Variable(tf.zeros([resout_dim]), trainable=True)

# 超参数
learning_rate = 0.001
epochs = 200
batch_size = 300

# 定义拉格朗日乘子
lambda1 = tf.Variable(0.3, trainable=False)
lambda2 = tf.Variable(0.3, trainable=False)
lambda3 = tf.Variable(0.3, trainable=False)
lambda4 = tf.Variable(0.3, trainable=False)
lambda5 = tf.Variable(0.3, trainable=False)
lambda6 = tf.Variable(0.3, trainable=False)
lambda7 = tf.Variable(0.3, trainable=False)
lambda8 = tf.Variable(0.3, trainable=False)
lambda9 = tf.Variable(0.3, trainable=False)
lambda10 = tf.Variable(0.3, trainable=False)

lambda1_re = tf.Variable(0.3, trainable=False)
lambda2_re = tf.Variable(0.3, trainable=False)
lambda3_re = tf.Variable(0.3, trainable=False)
lambda4_re = tf.Variable(0.3, trainable=False)
lambda5_re = tf.Variable(0.3, trainable=False)
lambda6_re = tf.Variable(0.3, trainable=False)
lambda7_re = tf.Variable(0.3, trainable=False)
lambda8_re = tf.Variable(0.3, trainable=False)
lambda9_re = tf.Variable(0.3, trainable=False)
lambda10_re = tf.Variable(0.3, trainable=False)

lambda1_vdn1 = tf.Variable(0.3, trainable=False)
lambda2_vdn1 = tf.Variable(0.3, trainable=False)
lambda3_vdn1 = tf.Variable(0.3, trainable=False)

lambda1_vdn2 = tf.Variable(0.3, trainable=False)
lambda2_vdn2 = tf.Variable(0.3, trainable=False)
lambda3_vdn2 = tf.Variable(0.3, trainable=False)

lambda1_vdn3 = tf.Variable(0.3, trainable=False)
lambda2_vdn3 = tf.Variable(0.3, trainable=False)
lambda3_vdn3 = tf.Variable(0.3, trainable=False)

lambda1_vdn4 = tf.Variable(0.3, trainable=False)
lambda2_vdn4 = tf.Variable(0.3, trainable=False)
lambda3_vdn4 = tf.Variable(0.3, trainable=False)

lambda1_vdn5 = tf.Variable(0.3, trainable=False)
lambda2_vdn5 = tf.Variable(0.3, trainable=False)
lambda3_vdn5 = tf.Variable(0.3, trainable=False)

lambda1_vdn1_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn1_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn1_2 = tf.Variable(0.3, trainable=False)

lambda1_vdn2_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn2_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn2_2 = tf.Variable(0.3, trainable=False)

lambda1_vdn3_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn3_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn3_2 = tf.Variable(0.3, trainable=False)

lambda1_vdn4_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn4_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn4_2 = tf.Variable(0.3, trainable=False)

lambda1_vdn5_2 = tf.Variable(0.3, trainable=False)
lambda2_vdn5_2 = tf.Variable(0.3, trainable=False)
lambda3_vdn5_2 = tf.Variable(0.3, trainable=False)

# 使用Adam优化器
optimizer = tf.optimizers.Adam(learning_rate)
optimizer1 = tf.optimizers.Adam(learning_rate)
optimizer_1 = tf.optimizers.Adam(learning_rate)
optimizer2 = tf.optimizers.Adam(learning_rate)
optimizer3 = tf.optimizers.Adam(learning_rate)
optimizer4 = tf.optimizers.Adam(learning_rate)
optimizer5 = tf.optimizers.Adam(learning_rate)
optimizer6 = tf.optimizers.Adam(learning_rate)
optimizer7 = tf.optimizers.Adam(learning_rate)
optimizer8 = tf.optimizers.Adam(learning_rate)
optimizer9 = tf.optimizers.Adam(learning_rate)
optimizer10 = tf.optimizers.Adam(learning_rate)
optimizer11 = tf.optimizers.Adam(learning_rate)
optimizer12 = tf.optimizers.Adam(learning_rate)
optimizer13 = tf.optimizers.Adam(learning_rate)
optimizer14 = tf.optimizers.Adam(learning_rate)
optimizer15 = tf.optimizers.Adam(learning_rate)
optimizer16 = tf.optimizers.Adam(learning_rate)


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

x_ex = round(random.uniform(0.2, 0.9), 22)
y_ex = round(random.uniform(0.2, 0.6), 22)
beta1_ex = round(random.uniform(0, 1), 22)
beta2_ex = round(random.uniform(0, 1), 22)
z_ex = round(random.uniform(0.2, 0.9), 22)
w1,g1,hg2 = delay(x_ex, y_ex, beta1_ex, z_ex)
w2,gg1,gg2 = delay(x_ex, y_ex, beta2_ex, 1-z_ex)
t_ex = 1/2*(w1+w2)
# for i in np.arange(120):
#     for j in np.arange(x_ex, 0.9, 0.1):
#         for k in np.arange(y_ex, 0.6, 0.1):
#              for m in np.arange(beta1_ex, 1, 0.1):
#                  for n in np.arange(beta2_ex, 1, 0.1):
#                      for z in np.arange(z_ex, 0.8, 0.1):
#                          t1, ggg1, ggg2 = delay(j, k, m, z)
#                          t2, gggg1, gggg2 = delay(j, k, n, 1-z)
#                          t = 1/2 * (t1+t2)
#                          if t<t_ex and not np.isnan(t) and t>0:
#                              t_ex = t
#                              x_ex1 = j
#                              y_ex1 = k
#                              beta1_ex1 = m
#                              beta2_ex1 = n
#                              z_ex1 = z
#     t_ex_points.append(float(t_ex))
#     t_ex_points1.append(float(t_ex))

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
            p12 = output[:, 1]
            p13 = output[:, 2]
            p14 = output[:, 3]
            p15 = output[:, 4]
            p21 = output[:, 5]
            p22 = output[:, 6]
            p23 = output[:, 7]
            p24 = 0.2 + 0.2 * output[:, 8]
            p25 = 0.2 + 0.2 * output[:, 9]
            p31 = output[:, 10]
            p32 = output[:, 11]
            p33 = output[:, 12]
            p34 = output[:, 13]
            p35 = output[:, 14]

            p4_cat_logits = tf.stack([output[:, 15], output[:, 16], output[:, 17], output[:, 18], output[:, 19]], axis=-1)
            p4_cat_probs = tf.nn.softmax(p4_cat_logits, axis=-1)  # Shape: (batch_size, 4)
            p41 = p4_cat_probs[:, 0]
            p42 = p4_cat_probs[:, 1]
            p43 = p4_cat_probs[:, 2]
            p44 = p4_cat_probs[:, 3]
            p45 = p4_cat_probs[:, 3]

            x1 = tf.reduce_mean(p11)
            x2 = tf.reduce_mean(p12)
            x3 = tf.reduce_mean(p13)
            x4 = tf.reduce_mean(p14)
            x5 = tf.reduce_mean(p15)
            y1 = tf.reduce_mean(p21)
            y2 = tf.reduce_mean(p22)
            y3 = tf.reduce_mean(p23)
            y4 = tf.reduce_mean(p24)
            y5 = tf.reduce_mean(p25)
            beta1 = tf.reduce_mean(p31)
            beta2 = tf.reduce_mean(p32)
            beta3 = tf.reduce_mean(p33)
            beta4 = tf.reduce_mean(p34)
            beta5 = tf.reduce_mean(p35)
            z1 = tf.reduce_mean(p41)
            z2 = tf.reduce_mean(p42)
            z3 = tf.reduce_mean(p43)
            z4 = tf.reduce_mean(p44)
            z5 = tf.reduce_mean(p45)
            # user1
            t1,w1,wk1 = delay(x1,y1,beta1,z1)
            # user2
            t2,w2,wk2 = delay(x2, y2, beta2, z2)
            # user3
            t3, w3, wk3 = delay(x3, y3, beta3, z3)
            # user4
            t4, w4, wk4 = delay(x4, y4, beta4, z4)
            # user5
            t5, w5, wk5 = delay(x5, y5, beta5, z5)
            # E2E delay
            w = 1 / 5 * (t1 + t2 + t3 + t4 + t5)
            MI1 = 0.05 * N * bs1 * np.log2(1 + x1 * pmax / sigmas)
            MI2 = 0.05 * N * bs2 * np.log2(1 + x2 * pmax / sigmas)
            MI3 = 0.05 * N * bs3 * np.log2(1 + x3 * pmax / sigmas)
            MI4 = 0.05 * N * bs4 * np.log2(1 + x4 * pmax / sigmas)
            MI5 = 0.05 * N * bs5 * np.log2(1 + x5 * pmax / sigmas)
            pmax = tf.constant(pmax, dtype=tf.float32)
            L = w + (lambda1 * (reta1 - MI1) + lambda2 * (reta2 - MI2) + lambda3 * (reta3 - MI3) + lambda4 * (reta4 - MI4) + lambda5 * (reta5 - MI5)) + (
                    lambda6 * (t1 - wmax1) + lambda7 * (t2 - wmax2) + lambda8 * (t3 - wmax3) + lambda9 * (t4 - wmax4) + lambda10 * (t5 - wmax5))

        with tf.GradientTape() as tape_1:
            # 前向传播
            h1_re = tf.nn.relu(tf.matmul(batch_X, W1_re) + b1_re)
            h2_re = tf.nn.relu(tf.matmul(h1_re, W2_re) + b2_re)
            h3_re = tf.nn.relu(tf.matmul(h2_re, W3_re) + b3_re)
            output_re = tf.nn.sigmoid(tf.matmul(h3_re, W4_re) + b4_re)  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_re = output_re[:, 0]
            p12_re = output_re[:, 1]
            p13_re = output_re[:, 2]
            p14_re = output_re[:, 3]
            p15_re = output_re[:, 4]
            p21_re = output_re[:, 5]
            p22_re = output_re[:, 6]
            p23_re = output_re[:, 7]
            p24_re = 0.2 + 0.2 * output_re[:, 8]
            p25_re = 0.2 + 0.2 * output_re[:, 9]
            p31_re = output_re[:, 10]
            p32_re = output_re[:, 11]
            p33_re = output_re[:, 12]
            p34_re = output_re[:, 13]
            p35_re = output_re[:, 14]

            p4_cat_logits_re = tf.stack([output_re[:, 15], output_re[:, 16], output_re[:, 17], output_re[:, 18],output_re[:, 19]], axis=-1)
            p4_cat_probs_re = tf.nn.softmax(p4_cat_logits_re, axis=-1)  # Shape: (batch_size, 4)
            p41_re = p4_cat_probs_re[:, 0]
            p42_re = p4_cat_probs_re[:, 1]
            p43_re = p4_cat_probs_re[:, 2]
            p44_re = p4_cat_probs_re[:, 3]
            p45_re = p4_cat_probs_re[:, 4]

            x1_re = tf.reduce_mean(p11_re)
            x2_re = tf.reduce_mean(p12_re)
            x3_re = tf.reduce_mean(p13_re)
            x4_re = tf.reduce_mean(p14_re)
            x5_re = tf.reduce_mean(p15_re)
            y1_re = tf.reduce_mean(p21_re)
            y2_re = tf.reduce_mean(p22_re)
            y3_re = tf.reduce_mean(p23_re)
            y4_re = tf.reduce_mean(p24_re)
            y5_re = tf.reduce_mean(p25_re)
            beta1_re = tf.reduce_mean(p31_re)
            beta2_re = tf.reduce_mean(p32_re)
            beta3_re = tf.reduce_mean(p33_re)
            beta4_re = tf.reduce_mean(p34_re)
            beta5_re = tf.reduce_mean(p35_re)
            z1_re = tf.reduce_mean(p41_re)
            z2_re = tf.reduce_mean(p42_re)
            z3_re = tf.reduce_mean(p43_re)
            z4_re = tf.reduce_mean(p44_re)
            z5_re = tf.reduce_mean(p45_re)
            # user1
            t1_re, w1_re, wk1_re = delay(x1_re, y1_re, beta1_re, z1_re)
            # user2
            t2_re, w2_re, wk2_re = delay(x2_re, y2_re, beta2_re, z2_re)
            # user3
            t3_re, w3_re, wk3_re = delay(x3_re, y3_re, beta3_re, z3_re)
            # user4
            t4_re, w4_re, wk4_re = delay(x4_re, y4_re, beta4_re, z4_re)
            # user5
            t5_re, w5_re, wk5_re = delay(x5_re, y5_re, beta5_re, z5_re)
            w_re = 1 / 5 * (t1_re + t2_re + t3_re + t4_re + t5_re)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # CRB1 = 0.01/x1/pmax
            # CRB2 = 0.01 / x2 / pmax
            MI1_re = 0.05 * bs1 * np.log2(1 + x1_re * pmax / sigmas)
            MI2_re = 0.05 * bs1 * np.log2(1 + x2_re * pmax / sigmas)
            MI3_re = 0.05 * bs3 * np.log2(1 + x3_re * pmax / sigmas)
            MI4_re = 0.05 * bs4 * np.log2(1 + x4_re * pmax / sigmas)
            MI5_re = 0.05 * bs5 * np.log2(1 + x5_re * pmax / sigmas)
            # CRB1_re = (ga + gb) / (x1_re * pmax * (ga * gb - gc * gc))
            # CRB2_re = (ga + gb) / (x2_re * pmax * (ga * gb - gc * gc))
            tt = math.pow(lambda1_re * (MI1_re - reta1), 2) + math.pow(lambda2_re * (MI2_re - reta2), 2) + math.pow(lambda3_re * (MI3_re - reta3), 2) + math.pow(lambda4_re * (MI4_re - reta4), 2)+ math.pow(lambda5_re * (MI5_re - reta5), 2)+ math.pow(lambda6_re * (t1_re - wmax1), 2)+ math.pow(lambda7_re * (t2_re - wmax2), 2)+ math.pow(lambda8_re * (t3_re - wmax3), 2)+ math.pow(lambda9_re * (t4_re - wmax4), 2)+ math.pow(lambda10_re * (t5_re - wmax5), 2)
            L_re = w_re + (lambda1_re * (reta1 - MI1_re) + lambda2_re * (reta2 - MI2_re) + lambda3_re * (reta3 - MI3_re) + lambda4_re * (reta4 - MI4_re)+lambda5_re * (reta5 - MI5_re)) + (
                    lambda6_re * (t1_re - wmax1) + lambda7_re * (t2_re - wmax2)+lambda8_re * (t3_re - wmax3) + lambda9_re * (t4_re - wmax4)+ lambda10_re * (t5_re - wmax5)) + 0.5 * tt

        with tf.GradientTape() as tape1:
            # 前向传播
            h1_vdn1 = tf.nn.relu(tf.matmul(batch_X, W11) + b11)
            h2_vdn1 = tf.nn.relu(tf.matmul(h1_vdn1, W21) + b21)
            h3_vdn1 = tf.nn.relu(tf.matmul(h2_vdn1, W31) + b31)
            output_vdn1 = tf.nn.sigmoid(tf.matmul(h3_vdn1, W41 + b41))  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn1 = output_vdn1[:, 0]
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
            A_vdn1 = 0.05 *  bs1 * np.log2(1 + p_vdn1 * pmax / sigmas)
            L_vdn1 = wk_vdn1 + lambda1_vdn1 * (reta1 - A_vdn1) + lambda2_vdn1 * (wk_vdn1 - wmax1)


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
            z_vdn2 = tf.reduce_mean(p51_vdn2)
            t_vdn2, w_vdn2, wk_vdn2 = delay(p_vdn2, f_vdn2, beta1_vdn2, z_vdn2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn2 = (ga + gb) / (p_vdn2 * pmax * (ga * gb - gc * gc))
            A_vdn2 = 0.05 * bs1 * np.log2(1 + p_vdn2 * pmax / sigmas)
            L_vdn2 = wk_vdn2 + lambda1_vdn2 * (reta2 - A_vdn2) + lambda2_vdn2 * (wk_vdn2 - wmax2)

        with tf.GradientTape() as tape3:
            # 前向传播
            h1_vdn3 = tf.nn.relu(tf.matmul(batch_X, W13) + b13)
            h2_vdn3 = tf.nn.relu(tf.matmul(h1_vdn3, W23) + b23)
            h3_vdn3 = tf.nn.relu(tf.matmul(h2_vdn3, W33) + b33)
            output_vdn3 = tf.nn.sigmoid(tf.matmul(h3_vdn3, W43 + b43))  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn3 = 0.2 + 0.2 * output_vdn3[:, 0]
            p21_vdn3 = output_vdn3[:, 1]
            p31_vdn3 = output_vdn3[:, 2]
            p41_vdn3 = output_vdn3[:, 3]
            p51_vdn3 = output_vdn3[:, 4]
            p_vdn3 = tf.reduce_mean(p11_vdn3)
            f_vdn3 = tf.reduce_mean(p21_vdn3)
            beta1_vdn3 = tf.reduce_mean(p31_vdn3)
            z_vdn3 = tf.reduce_mean(p51_vdn3)

            t_vdn3, w_vdn3, wk_vdn3 = delay(p_vdn3, f_vdn3, beta1_vdn3, z_vdn3)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn1 = (ga + gb) / (p_vdn1 * pmax * (ga * gb - gc * gc))
            A_vdn3 = 0.05 * bs3 * np.log2(1 + p_vdn3 * pmax / sigmas)
            L_vdn3 = wk_vdn3 + lambda1_vdn3 * (reta3 - A_vdn3) + lambda2_vdn3 * (wk_vdn3 - wmax3)

        with tf.GradientTape() as tape4:
            # 前向传播
            h1_vdn4 = 0.2 + 0.2 * tf.nn.relu(tf.matmul(batch_X, W14) + b14)
            h2_vdn4 = tf.nn.relu(tf.matmul(h1_vdn4, W24) + b24)
            h3_vdn4 = tf.nn.relu(tf.matmul(h2_vdn4, W34) + b34)
            output_vdn4 = tf.nn.sigmoid(tf.matmul(h3_vdn4, W44 + b44))  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn4 = output_vdn4[:, 0]
            p21_vdn4 = output_vdn4[:, 1]
            p31_vdn4 = output_vdn4[:, 2]
            p41_vdn4 = output_vdn4[:, 3]
            p51_vdn4 = output_vdn4[:, 4]
            p_vdn4 = tf.reduce_mean(p11_vdn4)
            f_vdn4 = tf.reduce_mean(p21_vdn4)
            beta1_vdn4 = tf.reduce_mean(p31_vdn4)
            z_vdn4 = tf.reduce_mean(p51_vdn4)

            t_vdn4, w_vdn4, wk_vdn4 = delay(p_vdn4, f_vdn4, beta1_vdn4, z_vdn4)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn1 = (ga + gb) / (p_vdn1 * pmax * (ga * gb - gc * gc))
            A_vdn4 = 0.05 * bs4 * np.log2(1 + p_vdn4 * pmax / sigmas)
            L_vdn4 = wk_vdn4 + lambda1_vdn4 * (reta4 - A_vdn4) + lambda2_vdn4 * (wk_vdn4 - wmax4)

        with tf.GradientTape() as tape5:
            # 前向传播
            h1_vdn5 = tf.nn.relu(tf.matmul(batch_X, W15) + b15)
            h2_vdn5 = tf.nn.relu(tf.matmul(h1_vdn5, W25) + b25)
            h3_vdn5 = tf.nn.relu(tf.matmul(h2_vdn5, W35) + b35)
            output_vdn5 = tf.nn.sigmoid(tf.matmul(h3_vdn5, W45 + b45))  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn5 = output_vdn5[:, 0]
            p21_vdn5 = 0.2 + 0.2 * output_vdn5[:, 1]
            p31_vdn5 = output_vdn5[:, 2]
            p41_vdn5 = output_vdn5[:, 3]
            p51_vdn5 = output_vdn5[:, 4]
            p_vdn5 = tf.reduce_mean(p11_vdn5)
            f_vdn5 = tf.reduce_mean(p21_vdn5)
            beta1_vdn5 = tf.reduce_mean(p31_vdn5)
            z_vdn5 = tf.reduce_mean(p51_vdn5)

            t_vdn5, w_vdn5, wk_vdn5 = delay(p_vdn5, f_vdn5, beta1_vdn5, z_vdn5)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn1 = (ga + gb) / (p_vdn1 * pmax * (ga * gb - gc * gc))
            A_vdn5 = 0.05 * bs5 * np.log2(1 + p_vdn5 * pmax / sigmas)
            L_vdn5 = wk_vdn5 + lambda1_vdn5 * (reta5 - A_vdn5) + lambda2_vdn5 * (wk_vdn5 - wmax5)

        with tf.GradientTape() as tape6:
            # 前向传播
            h1_vdn1_2 = tf.nn.relu(tf.matmul(batch_X, W11_2) + b11_2)
            h2_vdn1_2 = tf.nn.relu(tf.matmul(h1_vdn1_2, W21_2) + b21_2)
            h3_vdn1_2 = tf.nn.relu(tf.matmul(h2_vdn1_2, W31_2) + b31_2)
            output_vdn1_2 = tf.nn.sigmoid(tf.matmul(h3_vdn1_2, W41_2 + b41_2))  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn1_2 = 0.2 + 0.2 * output_vdn1_2[:, 0]
            p21_vdn1_2 = output_vdn1_2[:, 1]
            p31_vdn1_2 = output_vdn1_2[:, 2]
            p41_vdn1_2 = output_vdn1_2[:, 3]
            p51_vdn1_2 = output_vdn1_2[:, 4]
            p_vdn1_2 = tf.reduce_mean(p11_vdn1_2)
            f_vdn1_2 = tf.reduce_mean(p21_vdn1_2)
            beta2_vdn1_2 = tf.reduce_mean(p41_vdn1_2)
            z_vdn1_2 = tf.reduce_mean(p51_vdn1_2)
            t_vdn1_2, w_vdn1_2, wk_vdn1_2 = delay(p_vdn1_2, f_vdn1_2, beta2_vdn1_2, z_vdn1_2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn1_2 = (ga + gb) / (p_vdn1_2 * pmax * (ga * gb - gc * gc))
            A_vdn1_2 = 0.05 *  bs1 * np.log2(1 + p_vdn1_2 * pmax / sigmas)
            L_vdn1_2 = w_vdn1_2 + lambda1_vdn1_2 * (reta1 - A_vdn1_2) + lambda2_vdn1_2 * (
                        w_vdn1_2 - wmax1)

        with tf.GradientTape() as tape7:
            # 前向传播
            h1_vdn2_2 = tf.nn.relu(tf.matmul(batch_X, W12_2) + b12_2)
            h2_vdn2_2 = tf.nn.relu(tf.matmul(h1_vdn2_2, W22_2) + b22_2)
            h3_vdn2_2 = tf.nn.relu(tf.matmul(h2_vdn2_2, W32_2) + b32_2)
            output_vdn2_2 = tf.nn.sigmoid(tf.matmul(h3_vdn2_2, W42_2) + b42_2)  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn2_2 = output_vdn2_2[:, 0]
            p21_vdn2_2 = 0.2 + 0.2 * output_vdn2_2[:, 1]
            p31_vdn2_2 = output_vdn2_2[:, 2]
            p41_vdn2_2 = output_vdn2_2[:, 3]
            p51_vdn2_2 = output_vdn2_2[:, 4]
            p_vdn2_2 = tf.reduce_mean(p11_vdn2_2)
            f_vdn2_2 = tf.reduce_mean(p21_vdn2_2)
            beta1_vdn2_2 = tf.reduce_mean(p31_vdn2_2)
            beta2_vdn2_2 = tf.reduce_mean(p41_vdn2_2)
            z_vdn2_2 = tf.reduce_mean(p51_vdn2_2)
            t_vdn2_2, w_vdn2_2, wk_vdn2_2 = delay(p_vdn2_2, f_vdn2_2, beta2_vdn1_2, z_vdn2_2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn2_2 = (ga + gb) / (p_vdn2_2 * pmax * (ga * gb - gc * gc))
            A_vdn2_2 = 0.05 *  bs1 * np.log2(1 + p_vdn2_2 * pmax / sigmas)
            L_vdn2_2 = w_vdn2_2 + lambda1_vdn2_2 * (reta2 - A_vdn2_2) + lambda2_vdn2_2 * (
                        w_vdn2_2 - wmax2)

        with tf.GradientTape() as tape8:
            # 前向传播
            h1_vdn3_2 = tf.nn.relu(tf.matmul(batch_X, W13_2) + b13_2)
            h2_vdn3_2 = tf.nn.relu(tf.matmul(h1_vdn3_2, W23_2) + b23_2)
            h3_vdn3_2 = tf.nn.relu(tf.matmul(h2_vdn3_2, W33_2) + b33_2)
            output_vdn3_2 = tf.nn.sigmoid(tf.matmul(h3_vdn3_2, W43_2) + b43_2)  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn3_2 = output_vdn3_2[:, 0]
            p21_vdn3_2 = output_vdn3_2[:, 1]
            p31_vdn3_2 = output_vdn3_2[:, 2]
            p41_vdn3_2 = output_vdn3_2[:, 3]
            p51_vdn3_2 = output_vdn3_2[:, 4]
            p_vdn3_2 = tf.reduce_mean(p11_vdn3_2)
            f_vdn3_2 = tf.reduce_mean(p21_vdn3_2)
            beta1_vdn3_2 = tf.reduce_mean(p31_vdn3_2)
            beta2_vdn3_2 = tf.reduce_mean(p41_vdn3_2)
            z_vdn3_2 = tf.reduce_mean(p51_vdn2_2)
            t_vdn3_2, w_vdn3_2, wk_vdn3_2 = delay(p_vdn3_2, f_vdn3_2, beta2_vdn3_2, z_vdn3_2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn2_2 = (ga + gb) / (p_vdn2_2 * pmax * (ga * gb - gc * gc))
            A_vdn3_2 = 0.05 *  bs3 * np.log2(1 + p_vdn3_2 * pmax / sigmas)
            L_vdn3_2 = w_vdn3_2 + lambda1_vdn3_2 * (reta3 - A_vdn3_2) + lambda2_vdn3_2 * (
                        w_vdn3_2 - wmax3)

        with tf.GradientTape() as tape9:
            # 前向传播
            h1_vdn4_2 = tf.nn.relu(tf.matmul(batch_X, W14_2) + b14_2)
            h2_vdn4_2 = tf.nn.relu(tf.matmul(h1_vdn2_2, W24_2) + b24_2)
            h3_vdn4_2 = tf.nn.relu(tf.matmul(h2_vdn2_2, W34_2) + b34_2)
            output_vdn4_2 = tf.nn.sigmoid(tf.matmul(h3_vdn4_2, W44_2) + b44_2)  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn4_2 = output_vdn4_2[:, 0]
            p21_vdn4_2 = output_vdn4_2[:, 1]
            p31_vdn4_2 = output_vdn4_2[:, 2]
            p41_vdn4_2 = output_vdn4_2[:, 3]
            p51_vdn4_2 = output_vdn4_2[:, 4]
            p_vdn4_2 = tf.reduce_mean(p11_vdn4_2)
            f_vdn4_2 = tf.reduce_mean(p21_vdn4_2)
            beta1_vdn4_2 = tf.reduce_mean(p31_vdn4_2)
            beta2_vdn4_2 = tf.reduce_mean(p41_vdn4_2)
            z_vdn4_2 = tf.reduce_mean(p51_vdn4_2)
            t_vdn4_2, w_vdn4_2, wk_vdn4_2 = delay(p_vdn4_2, f_vdn4_2, beta2_vdn4_2, z_vdn4_2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn2_2 = (ga + gb) / (p_vdn2_2 * pmax * (ga * gb - gc * gc))
            A_vdn4_2 = 0.05 * bs1 * np.log2(1 + p_vdn4_2 * pmax / sigmas)
            L_vdn4_2 = w_vdn4_2 + lambda1_vdn4_2 * (reta4 - A_vdn4_2) + lambda2_vdn4_2 * (
                    w_vdn4_2 - wmax4)

        with tf.GradientTape() as tape10:
            # 前向传播
            h1_vdn5_2 = tf.nn.relu(tf.matmul(batch_X, W15_2) + b15_2)
            h2_vdn5_2 = tf.nn.relu(tf.matmul(h1_vdn5_2, W25_2) + b25_2)
            h3_vdn5_2 = tf.nn.relu(tf.matmul(h2_vdn5_2, W35_2) + b35_2)
            output_vdn5_2 = tf.nn.sigmoid(tf.matmul(h3_vdn5_2, W45_2) + b45_2)  # 使用 sigmoid 确保输出在 0 到 1 之间
            # 提取功率、计算频率和卸载策略
            p11_vdn5_2 = output_vdn5_2[:, 0]
            p21_vdn5_2 = output_vdn5_2[:, 1]
            p31_vdn5_2 = output_vdn5_2[:, 2]
            p41_vdn5_2 = output_vdn5_2[:, 3]
            p51_vdn5_2 = output_vdn5_2[:, 4]
            p_vdn5_2 = tf.reduce_mean(p11_vdn5_2)
            f_vdn5_2 = tf.reduce_mean(p21_vdn5_2)
            beta1_vdn5_2 = tf.reduce_mean(p31_vdn5_2)
            beta2_vdn5_2 = tf.reduce_mean(p41_vdn5_2)
            z_vdn5_2 = tf.reduce_mean(p51_vdn5_2)
            t_vdn5_2, w_vdn5_2, wk_vdn5_2 = delay(p_vdn5_2, f_vdn5_2, beta2_vdn5_2, z_vdn5_2)
            pmax = tf.constant(pmax, dtype=tf.float32)
            # CRB
            # A_vdn2_2 = (ga + gb) / (p_vdn2_2 * pmax * (ga * gb - gc * gc))
            A_vdn5_2 = 0.05 * bs5 * np.log2(1 + p_vdn5_2 * pmax / sigmas)
            L_vdn5_2 = w_vdn5_2 + lambda1_vdn5_2 * (reta5 - A_vdn5_2) + lambda2_vdn5_2 * (
                    w_vdn5_2 - wmax5)
        # t_c1 = (1 - beta1_vdn1) *  w_vdn1 + beta1_vdn1 * w_vdn2
        # t_c2 = (1 - beta2_vdn2) * w_vdn1 + beta2_vdn2 * w_vdn2
        # t_c3 = (1 - beta1_vdn1_2) *  w_vdn1 + beta1_vdn1_2 * w_vdn2
        # t_c4 = (1 - beta2_vdn2_2) * w_vdn1 + beta2_vdn2_2 * w_vdn2
        # t_de = 1/2 * (1/2 * (t_c1 + t_c2) + 1/2 * (t_c3 + t_c4))
        t_de = 1/5 * ((1 - beta1_vdn1) * wk_vdn1 + beta1_vdn1 * w_vdn1_2 + (1 - beta1_vdn2) * wk_vdn2 + beta1_vdn2 * w_vdn2_2 + (1 - beta1_vdn3) * wk_vdn3 + beta1_vdn3 * w_vdn3_2 + (1 - beta1_vdn4) * wk_vdn4 + beta1_vdn4 * w_vdn4_2 + (1 - beta1_vdn5) * wk_vdn5 + beta1_vdn5 * w_vdn5_2)
        # 残差连接
        with tf.GradientTape() as tape11:
            t_res = tf.stack([wk_vdn1, w_vdn1_2])
            t_res = tf.reshape(t_res, [2, 1])
            res1 = tf.nn.relu(tf.matmul(t_res, res_w1) + res_b1)
            res2 = tf.nn.relu(tf.matmul(res1, res_w2) + res_b2)
            res3 = 0.11 * tf.nn.tanh(tf.matmul(res2, res_w3) + res_b3)
            t_with_res = tf.nn.relu(t_res + res3)

        with tf.GradientTape() as tape12:
            t_res_2 = tf.stack([wk_vdn2, w_vdn2_2])
            t_res_2 = tf.reshape(t_res_2, [2, 1])
            res1_2 = tf.nn.relu(tf.matmul(t_res_2, res_w1_2) + res_b1_2)
            res2_2 = tf.nn.relu(tf.matmul(res1_2, res_w2_2) + res_b2_2)
            res3_2 = 0.11 * tf.nn.tanh(tf.matmul(res2_2, res_w3_2) + res_b3_2)
            t_with_res_2 = tf.nn.relu(t_res_2 + res3_2)

        with tf.GradientTape() as tape13:
            t_res_3 = tf.stack([wk_vdn3, w_vdn3_2])
            t_res_3 = tf.reshape(t_res_3, [2, 1])
            res1_3 = tf.nn.relu(tf.matmul(t_res_3, res_w1_3) + res_b1_3)
            res2_3 = tf.nn.relu(tf.matmul(res1_3, res_w2_3) + res_b2_3)
            res3_3 = 0.11 * tf.nn.tanh(tf.matmul(res2_3, res_w3_3) + res_b3_3)
            t_with_res_3 = tf.nn.relu(t_res_3 + res3_3)

        with tf.GradientTape() as tape14:
            t_res_4 = tf.stack([wk_vdn4, w_vdn4_2])
            t_res_4 = tf.reshape(t_res_4, [2, 1])
            res1_4 = tf.nn.relu(tf.matmul(t_res_4, res_w1_4) + res_b1_4)
            res2_4 = tf.nn.relu(tf.matmul(res1_4, res_w2_4) + res_b2_4)
            res3_4 = 0.11 * tf.nn.tanh(tf.matmul(res2_4, res_w3_4) + res_b3_4)
            t_with_res_4 = tf.nn.relu(t_res_4 + res3_4)

        with tf.GradientTape() as tape15:
            t_res_5 = tf.stack([wk_vdn5, w_vdn5_2])
            t_res_5 = tf.reshape(t_res_5, [2, 1])
            res1_5 = tf.nn.relu(tf.matmul(t_res_5, res_w1_5) + res_b1_5)
            res2_5 = tf.nn.relu(tf.matmul(res1_5, res_w2_5) + res_b2_5)
            res3_5 = 0.11 * tf.nn.tanh(tf.matmul(res2_5, res_w3_5) + res_b3_5)
            t_with_res_5 = tf.nn.relu(t_res_5 + res3_5)

        t_total_rdn1_1 = (1 - beta1_vdn1) * t_with_res[0][0] + beta1_vdn1 * t_with_res[1][0]
        t_total_rdn1_2 = (1 - beta2_vdn1_2) * t_with_res[0][0] + beta2_vdn1_2 * t_with_res[1][0]
        t_total_rdn_1 = 1 / 2 * (t_total_rdn1_1 + t_total_rdn1_2)

        t_total_rdn2_1 = (1 - beta1_vdn2) * t_with_res_2[0][0] + beta1_vdn2 * t_with_res_2[1][0]
        t_total_rdn2_2 = (1 - beta2_vdn2_2) * t_with_res_2[0][0] + beta2_vdn2_2 * t_with_res_2[1][0]
        t_total_rdn_2 = 1 / 2 * (t_total_rdn2_1 + t_total_rdn2_2)

        t_total_rdn3_1 = (1 - beta1_vdn3) * t_with_res_3[0][0] + beta1_vdn3 * t_with_res_3[1][0]
        t_total_rdn3_2 = (1 - beta2_vdn3_2) * t_with_res_3[0][0] + beta2_vdn3_2 * t_with_res_3[1][0]
        t_total_rdn_3 = 1 / 2 * (t_total_rdn3_1 + t_total_rdn3_2)

        t_total_rdn4_1 = (1 - beta1_vdn4) * t_with_res_4[0][0] + beta1_vdn4 * t_with_res_4[1][0]
        t_total_rdn4_2 = (1 - beta2_vdn4_2) * t_with_res_4[0][0] + beta2_vdn4_2 * t_with_res_4[1][0]
        t_total_rdn_4 = 1 / 2 * (t_total_rdn4_1 + t_total_rdn4_2)

        t_total_rdn5_1 = (1 - beta1_vdn5) * t_with_res_5[0][0] + beta1_vdn5 * t_with_res_5[1][0]
        t_total_rdn5_2 = (1 - beta2_vdn5_2) * t_with_res_5[0][0] + beta2_vdn5_2 * t_with_res_5[1][0]
        t_total_rdn_5 = 1 / 2 * (t_total_rdn5_1 + t_total_rdn5_2)

        with tf.GradientTape() as tape16:
            t_res_total = tf.stack([t_total_rdn1_1, t_total_rdn2_1, t_total_rdn3_1, t_total_rdn4_1, t_total_rdn5_1])
            t_res_total = tf.reshape(t_res_total, [5, 1])
            res1_total = tf.nn.relu(tf.matmul(t_res_total, res_w1_total) + res_b1_total)
            res2_total = tf.nn.relu(tf.matmul(res1_total, res_w2_total) + res_b2_total)
            res3_total = 0.11 * tf.nn.tanh(tf.matmul(res2_total, res_w3_total) + res_b3_total)
            t_with_res_total = tf.nn.relu(t_res_total + res3_total)

        t_total_rdn = 1 / 5 * (t_with_res_total[0][0] + t_with_res_total[1][0] + t_with_res_total[2][0] +t_with_res_total[3][0]+t_with_res_total[4][0])

        # 参数更新
        gradients = tape.gradient(L, [W1, b1, W2, b2, W3, b3, W4, b4])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3, W4, b4]))
        # 更新拉格朗日乘子
        lambda1 = lambda1 - v5 * (reta1 - MI1)
        lambda2 = lambda2 - v6 * (reta2 - MI2)
        lambda3 = lambda3 - v5 * (reta3 - MI3)
        lambda4 = lambda4 - v6 * (reta4 - MI4)
        lambda5 = lambda5 - v6 * (reta5 - MI5)
        lambda6 = lambda6 - v7 * (w1 - wmax1)
        lambda7 = lambda7 - v8 * (w2 - wmax2)
        lambda8 = lambda8 - v7 * (w3 - wmax3)
        lambda9 = lambda9 - v8 * (w4 - wmax4)
        lambda10 = lambda10 - v8 * (w5 - wmax5)

        gradients__1 = tape_1.gradient(L_re, [W1_re, b1_re, W2_re, b2_re, W3_re, b3_re, W4_re, b4_re])
        optimizer_1.apply_gradients(zip(gradients__1, [W1_re, b1_re, W2_re, b2_re, W3_re, b3_re, W4_re, b4_re]))
        # 更新拉格朗日乘子
        lambda1_re = lambda1_re - v5 * (reta1 - MI1_re)
        lambda2_re = lambda2_re - v6 * (reta2 - MI2_re)
        lambda3_re = lambda3_re - v5 * (reta3 - MI3_re)
        lambda4_re = lambda4_re - v6 * (reta4 - MI4_re)
        lambda5_re = lambda5_re - v6 * (reta5 - MI5_re)
        lambda6_re = lambda6_re - v7 * (w1_re - wmax1)
        lambda7_re = lambda7_re - v8 * (w2_re - wmax2)
        lambda8_re = lambda8_re - v7 * (w3_re - wmax3)
        lambda9_re = lambda9_re - v8 * (w4_re - wmax4)
        lambda10_re = lambda10_re - v8 * (w5_re - wmax5)

        gradients_1 = tape1.gradient(L_vdn1, [W11, b11, W21, b21, W31, b31, W41, b41])
        optimizer1.apply_gradients(zip(gradients_1, [W11, b11, W21, b21, W31, b31, W41, b41]))
        lambda1_vdn1 = lambda1_vdn1 - v5 * (reta1 - A_vdn1)
        lambda2_vdn1 = lambda2_vdn1 - v7 * (wk_vdn1 - wmax1)

        gradients_2 = tape2.gradient(L_vdn2, [W12, b12, W22, b22, W32, b32, W42, b42])
        optimizer2.apply_gradients(zip(gradients_2, [W12, b12, W22, b22, W32, b32, W42, b42]))
        lambda1_vdn2 = lambda1_vdn2 - v5 * (reta1 - A_vdn2)
        lambda2_vdn2 = lambda2_vdn2 - v7 * (wk_vdn2 - wmax1)

        gradients_3 = tape3.gradient(L_vdn3, [W13, b13, W23, b23, W33, b33, W43, b43])
        optimizer3.apply_gradients(zip(gradients_3, [W13, b13, W23, b23, W33, b33, W43, b43]))
        lambda1_vdn3 = lambda1_vdn3 - v5 * (reta3 - A_vdn3)
        lambda2_vdn3 = lambda2_vdn3 - v7 * (wk_vdn3 - wmax3)

        gradients_4 = tape4.gradient(L_vdn4, [W14, b14, W24, b24, W34, b34, W44, b44])
        optimizer4.apply_gradients(zip(gradients_4, [W14, b14, W24, b24, W34, b34, W44, b44]))
        lambda1_vdn4 = lambda1_vdn4 - v5 * (reta4- A_vdn4)
        lambda2_vdn4 = lambda2_vdn4 - v7 * (wk_vdn4 - wmax4)

        gradients_5 = tape5.gradient(L_vdn5, [W15, b15, W25, b25, W35, b35, W45, b45])
        optimizer5.apply_gradients(zip(gradients_5, [W15, b15, W25, b25, W35, b35, W45, b45]))
        lambda1_vdn5 = lambda1_vdn5 - v5 * (reta5 - A_vdn5)
        lambda2_vdn5 = lambda2_vdn5 - v7 * (wk_vdn5 - wmax5)

        gradients_6 = tape6.gradient(L_vdn1_2, [W11_2, b11_2, W21_2, b21_2, W31_2, b31_2, W41_2, b41_2])
        optimizer6.apply_gradients(zip(gradients_6, [W11_2, b11_2, W21_2, b21_2, W31_2, b31_2, W41_2, b41_2]))
        lambda1_vdn1_2 = lambda1_vdn1_2 - v5 * (reta1 - A_vdn1_2)
        lambda2_vdn1_2 = lambda2_vdn1_2 - v7 * (w_vdn1_2 - wmax1)

        gradients_7 = tape7.gradient(L_vdn2_2, [W12_2, b12_2, W22_2, b22_2, W32_2, b32_2, W42_2, b42_2])
        optimizer7.apply_gradients(zip(gradients_7, [W12_2, b12_2, W22_2, b22_2, W32_2, b32_2, W42_2, b42_2]))
        lambda1_vdn2_2 = lambda1_vdn2_2 - v5 * (reta2 - A_vdn2_2)
        lambda2_vdn2_2 = lambda2_vdn2_2 - v7 * (w_vdn2_2 - wmax2)

        gradients_8 = tape8.gradient(L_vdn3_2, [W13_2, b13_2, W23_2, b23_2, W33_2, b33_2, W43_2, b43_2])
        optimizer8.apply_gradients(zip(gradients_8, [W13_2, b13_2, W23_2, b23_2, W33_2, b33_2, W43_2, b43_2]))
        lambda1_vdn3_2 = lambda1_vdn3_2 - v5 * (reta3 - A_vdn3_2)
        lambda2_vdn3_2 = lambda2_vdn3_2 - v7 * (w_vdn3_2 - wmax3)

        gradients_9 = tape9.gradient(L_vdn4_2, [W14_2, b14_2, W24_2, b24_2, W34_2, b34_2, W44_2, b44_2])
        optimizer9.apply_gradients(zip(gradients_9, [W14_2, b14_2, W24_2, b24_2, W34_2, b34_2, W44_2, b44_2]))
        lambda1_vdn4_2 = lambda1_vdn4_2 - v5 * (reta4 - A_vdn4_2)
        lambda2_vdn4_2 = lambda2_vdn4_2 - v7 * (w_vdn4_2 - wmax4)

        gradients_10 = tape10.gradient(L_vdn5_2, [W15_2, b15_2, W25_2, b25_2, W35_2, b35_2, W45_2, b45_2])
        optimizer10.apply_gradients(zip(gradients_10, [W15_2, b15_2, W25_2, b25_2, W35_2, b35_2, W45_2, b45_2]))
        lambda1_vdn5_2 = lambda1_vdn5_2 - v5 * (reta5 - A_vdn5_2)
        lambda2_vdn5_2 = lambda2_vdn5_2 - v7 * (w_vdn5_2 - wmax5)

        gradients_11 = tape11.gradient(t_with_res,
                                     [res_w1, res_b1, res_w2, res_b2, res_w3, res_b3])
        optimizer11.apply_gradients(zip(gradients_11,
                                       [res_w1, res_b1, res_w2, res_b2, res_w3, res_b3]))

        gradients_12 = tape12.gradient(t_with_res_2,
                                     [res_w1_2, res_b1_2, res_w2_2, res_b2_2, res_w3_2, res_b3_2])
        optimizer12.apply_gradients(zip(gradients_12,
                                       [res_w1_2, res_b1_2, res_w2_2, res_b2_2, res_w3_2, res_b3_2]))

        gradients_13 = tape13.gradient(t_with_res_3,
                                     [res_w1_3, res_b1_3, res_w2_3, res_b2_3, res_w3_3, res_b3_3])
        optimizer13.apply_gradients(zip(gradients_13,
                                       [res_w1_3, res_b1_3, res_w2_3, res_b2_3, res_w3_3, res_b3_3]))

        gradients_14 = tape14.gradient(t_with_res_4,
                                       [res_w1_4, res_b1_4, res_w2_4, res_b2_4, res_w3_4, res_b3_4])
        optimizer14.apply_gradients(zip(gradients_14,
                                        [res_w1_4, res_b1_4, res_w2_4, res_b2_4, res_w3_4, res_b3_4]))

        gradients_15 = tape15.gradient(t_with_res_5,
                                       [res_w1_5, res_b1_5, res_w2_5, res_b2_5, res_w3_5, res_b3_5])
        optimizer15.apply_gradients(zip(gradients_15,
                                        [res_w1_5, res_b1_5, res_w2_5, res_b2_5, res_w3_5, res_b3_5]))

        gradients_16 = tape16.gradient(t_with_res_total,
                                     [res_w1_total, res_b1_total, res_w2_total, res_b2_total, res_w3_total,
                                      res_b3_total])
        optimizer16.apply_gradients(zip(gradients_16,
                                       [res_w1_total, res_b1_total, res_w2_total, res_b2_total, res_w3_total,
                                        res_b3_total]))
    print(
        f"Epoch {epoch + 1}")

    points.append(epoch + 1)
    # if epoch == 50:
    #     print("4", float(1 / 2 * (p_vdn1 + p_vdn2)), float(1 / 2 * (p_vdn1_2 + p_vdn2_2)),
    #           float(1 / 2 * (f_vdn1 + f_vdn2)), float(1 / 2 * (f_vdn1_2 + f_vdn2_2)),
    #           float(beta1_vdn1), float(beta2_vdn1_2), float(z_vdn1))
    #     print("3", float(1 / 2 * (p_vdn1 + p_vdn2)), float(1 / 2 * (p_vdn1_2 + p_vdn2_2)),
    #           float(1 / 2 * (f_vdn1 + f_vdn2)), float(1 / 2 * (f_vdn1_2 + f_vdn2_2)),
    #           float(beta1_vdn1), float(beta2_vdn1_2), float(z_vdn1))
    #     print("3", float(x1_re), float(x2_re), float(y1_re), float(y2_re), float(beta1_re), float(beta2_re),
    #           float(z_re))
    #     print("2", float(x1), float(x2), float(y1), float(y2), float(beta1), float(beta2), float(z))


    t_points.append(float(w))
    t_re_points.append(float(w_re))
#     # t_re_points1.append(w_re)
    # w_de = 1 / 2 * ((1 - beta1_vdn1) * wk_vdn1 + beta1_vdn1 * w_vdn2 + (
    #         1 - beta2_vdn1_2) * wk_vdn1_2 + beta2_vdn1_2 * w_vdn2_2)
    t_1_points.append(float(t_de))
    # t_1_points.append(float(t_de) + np.random.uniform(-0.006, 0.005, 1))
    t_2_points.append(float(t_total_rdn))
#
#
# print("4", float(1 /2 * (p_vdn1 + p_vdn2 )), float(1/2 * (p_vdn1_2 + p_vdn2_2)), float(1 / 2 * (f_vdn1 + f_vdn2)), float(1 / 2 * (f_vdn1_2 + f_vdn2_2)),
#       float(beta1_vdn1), float(beta2_vdn1_2), float(z_vdn1))
# print("3", float(1 /2 * (p_vdn1 + p_vdn2)), float(1/2 * (p_vdn1_2 + p_vdn2_2)), float(1 / 2 * (f_vdn1 + f_vdn2)), float(1 / 2 * (f_vdn1_2 + f_vdn2_2)),
#       float(beta1_vdn1), float(beta2_vdn1_2), float(z_vdn1))
print("3", float(x1_re), float(x2_re), float(y1_re), float(y2_re), float(beta1_re), float(beta2_re), float(z1_re), float(z2_re), float(z3_re), float(z4_re))
print("2", float(x1), float(x2), float(y1), float(y2), float(beta1), float(beta2), float(z1),float(z2),float(z3))
# print("1", float(x_ex), float(y_ex), float(beta1_ex), float(beta2_ex), float(z_ex))

# plt.xlabel('Iterations')
# plt.ylabel('Lagrangian Function')
# plt.legend()
# df1 = DataFrame({'number': points, 'value2': t_points, 'value3': t_re_points, 'value4': t_1_points, 'value5': t_2_points})
# df1.to_excel('algorithm.xlsx',  index=False)
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
