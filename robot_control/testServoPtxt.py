# coding=utf-8
from CCClient import CCClient
import time

# 读取点位信息文件
fname = "servoPdata.txt"

# 读取每一行数据存进 teach 数组中
with open(fname, 'r+', encoding='utf-8') as f:
    teach = [i[:-1].split(',') for i in f.readlines()]

# print(teach)

# 连接机器人
cps = CCClient()  # 构造机器人库的类
cps.connectTCPSocket('192.168.11.7')  # 连接到 机器人的 IP 地址
cps.SetOverride(0.10)  # 设置速度比为：末端最大速度约束的 10%

# 运动到第一个位置
joints = teach[0][0:6]
print(cps.moveL(joints))

# 等待运动完成
cps.waitMoveDone()

startTime = time.perf_counter()
servoTime = 0.025  # 设置伺服周期为 25ms，建议最小不低于 15ms
# 设置前瞻时间，前瞻时间越大，轨迹越平滑，但越滞后。
lookaheadTime = 0.2  # 设置前瞻时间为 200ms,建议在 0.05s~0.2s 之间
# 开启在线控制(ServoP)
print(cps.startServo(servoTime, lookaheadTime))  # start Servo
i = 0
failTime = 0
loopTime = 1  # 循环运行多少次
flag = True  # 停止标志

# 调用 pushServoP 接口，在线伺服控制机器人
while (flag):
    loopCount = 0
    while flag:
        if i >= len(teach) * loopTime:
            flag = False  # 循环跑 loopTime 次，发完停止
            break
        teach[loopCount][6:18] = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']  # 加上 UCS,TCP 的坐标
        recv = (cps.pushServoP(teach[loopCount][0:18])).decode()  # update servo position
        # recv += "  " + 'Fail Times : ' + str(failTime)
        # print(recv)
        # recvData = recv.split(',')
        # if (recvData[1] != 'OK'):
        #     failTime = failTime + 1 # 多少次返回了失败

        # 确保按照伺服周期时间发送位置
        while True:  # waiting for next servo time
            currentTime = time.perf_counter()
            if (currentTime - startTime) > (servoTime * (loopCount + 1)):
                break
            time.sleep(0.0001)
        i += 1
        loopCount += 1
    end = time.perf_counter()
    print(end - startTime)  # 打印运行了多少时间
    print(len(teach))
    loopTime = loopTime + 1
