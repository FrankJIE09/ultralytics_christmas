# coding=utf-8
from CCClient import CCClient
import time
import csv


cps = CCClient()    # 构造机器人库的类
cps.connectTCPSocket('192.168.11.7')   # 连接到 机器人的 IP 地址
cps.SetOverride(0.50)   # 设置速度比为：关节最大速度约束的 50%

# 读取关节位置数据文件
readCSV = open('servoj_data.csv', 'r')
reader = csv.reader(readCSV)

# 将关节位置按行存到 teach 数组中
teach = []
for item in reader:
    teach.append(item)
del teach[0] # 删除前两行
del teach[0]

joints = teach[0][0:6]
print (cps.moveJ(joints))   # 运动到第一个位置
cps.waitMoveDone() # 等待运动完成
startTime = time.perf_counter()
servoTime = 0.025  # 设置伺服周期为 25ms，建议最小不低于 15ms
# 设置前瞻时间，前瞻时间越大，轨迹越平滑，但越滞后。
lookaheadTime = 0.2 # 设置前瞻时间为 200ms,建议在 0.05s~0.2s 之间
print (cps.startServo(servoTime, lookaheadTime))    # 开启在线伺服控制
count = 0
# 调用 pushServoJ 接口，在线伺服控制机器人
while True:
    if count > len(teach)-1:
        break   # 只跑一次，发完就停止
    cps.pushServoJ(teach[count][18:24]) # 发送关节位置给机器人
    # 确保按照伺服周期时间发送位置
    while True:
        currentTime = time.perf_counter()
        if (currentTime-startTime) > (servoTime * (count + 1)):
            break
        time.sleep(0.0001)
    count += 1
end = time.perf_counter()
print(end - startTime)  # 打印运行了多少时间
print(len(teach))
