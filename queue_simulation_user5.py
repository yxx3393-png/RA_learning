import simpy
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
E2E_delay = []
points = []
processing_delays = []
local_delays = []
processing_delays1 = []
local_delays1 = []
edge_comp_delays = []
edge_comp_delayss1 = []
com_delay1 = []
com_delayss1 = []
se_delayss1 = []
se_delay = []
E2E_delay1 = []
E2E_delay2 = []
total_com_ses = []
# 参数设置
np.random.seed(42)  # 固定随机种子
arrival_rate = 2   # 泊松到达速率 λ (任务/秒)
packet_size = 230000  # 数据包大小 (bits)
timeslot = 0.05  # 时隙长度 (秒)

# a = 0.3
b = 0.3
# 仿真时长
simulation_time = 10  # 单位：秒

for a in np.arange(0.2,0.81,0.04):
    # 服务速率 (bits/s)
    sensing_rate = 50000 * np.log(1 + a * 1.2 / 10e-12) / 0.05  # 第一个队列服务速率
    processing_rate = b * 20000000  # 第二个队列服务速率
    local_rate = (1 - b) * 20000000   # 第三个队列服务速率
    transmission_rate = 50000 * np.log(1 + (1 - a) * 1.2 / 10e-12) / 0.05  # 第三个队列服务速率
    edge_rate = 20000000 / 5   # 第四个队列的服务速率

    # 计算每时隙可处理的数据量
    def service_time_in_slots(rate):
        bits_per_slot = rate   # 每时隙处理的数据量
        slots_required = packet_size / bits_per_slot # 计算需要的时间
        return slots_required  # 返回总服务时间


    # 定义队列阶段
    def queue_stage(env, resource, service_duration):
        queue_start_time = env.now
        with resource.request() as req:
            yield req
            wait_time = env.now - queue_start_time  # 排队等待时间
            yield env.timeout(service_duration)  # 服务时间
        return wait_time + service_duration  # 返回总时延


    # 定义任务流
    def task_flow(env, arrival_rate, sensing, processing, local, transmission, edge, offload_prob):
        while True:
            # 生成下一个任务的到达时间，并对齐到时隙
            next_arrival = np.random.exponential(1 / arrival_rate)
            # next_arrival = np.ceil(next_arrival / timeslot) * timeslot
            yield env.timeout(next_arrival)
            arrival_time1 = env.now  # 记录到达时间
            # 计算本地路径时延 (T1)
            t1 = 0
            se_delay1 = yield env.process(queue_stage(env, sensing, sensing_duration))
            completion_time1 = env.now
            processing_delay1 = yield env.process(queue_stage(env, processing, processing_duration))
            completion_time2 = env.now
            local_delay = yield env.process(queue_stage(env, local, local_duration))
            completion_time3 = env.now
            t1 += processing_delay1 + local_delay
            # t1 += yield env.process(queue_stage(env, processing, processing_duration))
            # t1 += yield env.process(queue_stage(env, local, local_duration))

            # 计算边缘路径时延 (T2)
            t2 = 0
            arrival_time2 = env.now  # 记录到达时间
            se_delay2 = yield env.process(queue_stage(env, sensing, sensing_duration))
            completion_time11 = env.now
            processing_delay2 = yield env.process(queue_stage(env, processing, processing_duration))
            completion_time21 = env.now
            com_delay = yield env.process(queue_stage(env, transmission, transmission_duration))
            completion_time31 = env.now
            comp_delay = yield env.process(queue_stage(env, edge, edge_duration))
            completion_time41 = env.now
            t2 += processing_delay2 + com_delay + comp_delay
            # 计算综合端到端时延
            total_delay = offload_prob* t1 + (1-offload_prob) * t2
            delays.append((1-offload_prob)*(completion_time41-arrival_time2)+offload_prob*(completion_time3-arrival_time1))
            delays1.append(t1)
            delays2.append(t2)
            processing_delays.append((1-offload_prob)*processing_delay1+offload_prob*processing_delay2)
            local_delays.append(local_delay)
            edge_comp_delays.append(completion_time41-completion_time31)
            com_delay1.append(com_delay)
            se_delay.append((1-offload_prob)*se_delay1+offload_prob*se_delay2)

    # 性能监控：统计每个时隙的已完成任务数
    def monitor(env, interval):
        while True:
            print(f"时隙 {env.now:.2f}s: 当前已完成任务 {len(delays)} 个")
            yield env.timeout(interval)


    # 初始化服务时间
    sensing_duration = service_time_in_slots(sensing_rate)
    processing_duration = service_time_in_slots(processing_rate)
    local_duration = service_time_in_slots(local_rate)
    transmission_duration = service_time_in_slots(transmission_rate)
    edge_duration = service_time_in_slots(edge_rate)

    # 仿真环境初始化
    env = simpy.Environment()
    delays = []  # 存储每个任务的端到端时延
    delays1 = []
    delays2 = []

    # 定义资源（队列）
    sensing = simpy.Resource(env, capacity=1)
    processing = simpy.Resource(env, capacity=1)
    local = simpy.Resource(env, capacity=1)
    transmission = simpy.Resource(env, capacity=1)
    edge = simpy.Resource(env, capacity=1)

    # 启动任务流
    offload_prob = 0.2  # 卸载到边缘的概率
    env.process(task_flow(env, arrival_rate, sensing, processing, local, transmission, edge, offload_prob))

    # 启动性能监控
    env.process(monitor(env, timeslot))

    # 运行仿真
    env.run(until=simulation_time)

    # 结果分析
    average_delay = np.mean(delays)/0.05+1
    E2E_delay.append(average_delay)
    average_delay1 = np.mean(delays1)/0.05+1
    E2E_delay1.append(average_delay1)
    average_delay2 = np.mean(delays2)/0.05+1
    E2E_delay2.append(average_delay2)
    processing_delay1 = np.mean(processing_delays)/0.05
    processing_delays1.append(processing_delay1)
    local_delay1 = np.mean(local_delays)/0.05
    local_delays1.append(local_delay1)
    points.append(a)
    edge_comp_delayss = np.mean(edge_comp_delays)/0.05
    edge_comp_delayss1.append(edge_comp_delayss)
    se_delayss = np.mean(se_delay) / 0.05
    se_delayss1.append(se_delayss)
    com_delayss = np.mean(com_delay1) / 0.05
    com_delayss1.append(com_delayss)
    total_com_se = com_delayss+se_delayss
    total_com_ses.append(total_com_se)

df1 = DataFrame({'number': points, 'value1': se_delayss1, 'value2': com_delayss1,'value3': total_com_ses})
df1.to_excel('simulation_p1.xlsx',  index=False)
plt.figure()
# plt.plot(points, processing_delays1, color='indigo', label='Simulation')
# plt.plot(points, local_delays1, color='black', label='Simulation')
plt.plot(points, se_delayss1, color='red', label='Simulation')
plt.plot(points, com_delayss1, color='blue', label='Simulation')
plt.plot(points, total_com_ses, color='green', label='Simulation')
# plt.plot(points, edge_comp_delayss1, color='green', label='Simulation')
# plt.plot(points, E2E_delay, color='seagreen', label='Simulation')
# plt.plot(points, E2E_delay1, color='springgreen', label='Simulation')
# plt.plot(points, E2E_delay2, color='springgreen', label='Simulation')
plt.show()
# print("\n仿真结果:")
# print(f"仿真运行时间: {simulation_time}s")
# print(f"平均端到端时延: {average_delay:.4f}slot")
# print(f"完成任务总数: {len(delays)} 个")
