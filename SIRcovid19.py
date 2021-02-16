# %%

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

# %%

parameters = {
    'N': 328200000,       # 美国总人口
    "if_mask": 2,    # 戴口罩与否 0：都不戴，1：患者戴，2：都戴
    "N_contact": 3,       # 感染者每天可以接触的人数
    "infect_Pro": [0.7, 0.05, 0.015],  # 接触后传染概率
    "D": 14,              # 传染者可传染时间（恢复天数）
    "N_vax": [0, 1369921],          # 1369921 每日疫苗供应量
    "p": [0.94, 0.95],              # 疫苗有效率
    "I0": 27700000-8478384,       # 感染人数
    "R0": 8478384,        # 恢复人数
    "V0": 14077440        # 疫苗接种人数
}


class SIR_covid19():
    """
    N         : 美国总人口

    if_mask   : 戴口罩与否 0：都不戴，1：患者戴，2：都戴

    N_contact : 感染者每天可以接触的人数

    infect_Pro: 接触后传染概率

    D         : 传染者可传染时间（恢复天数）

    N_vax     : 每日疫苗供应量

    p         : 疫苗有效率

    beta      : β：被感染者每天可以感染人数

    gamma     : γ：康复比例

    vax       : 疫苗有效注射率
    """

    def __init__(self, N, if_mask, N_contact, infect_Pro, D, N_vax, p,
                 I0, R0, V0):
        self.N = N  # 美国总人口
        self.if_mask = if_mask  # 戴口罩比例
        self.N_contact = N_contact  # 感染者每天可以接触的人数
        self.infect_Pro = infect_Pro  # 接触后传染概率
        self.D = D  # 传染者可传染时间（恢复天数）
        self.N_vax = N_vax  # 每日疫苗供应量
        self.p = p  # 疫苗有效率

        # β：被感染者每天可以感染人数
        self.beta = N_contact * infect_Pro[if_mask]
        self.gamma = 1 / D  # γ：康复比例
        self.vax = (N_vax[0]*p[0] + N_vax[1]*p[1])/N   # 疫苗有效注射率

        # 初始条件：27700000人为感染，8478384人康复, 0人打疫苗
        self.I0, self.R0, self.V0 = I0, R0, V0
        self.S0 = N - I0 - R0 - V0

    def deriv(self, y, t, N, beta, gamma, vax):

        S, I, R, V = y

        dSdt = -beta * S * I / N - vax * S

        dVdt = vax * S

        dIdt = beta * S * I / N - gamma * I

        dRdt = gamma * I

        return dSdt, dIdt, dRdt, dVdt

    def run(self, t):
        """
        计算模型
        t: 所看时间
        """
        T = np.linspace(0, t, t)  # 时间作为x轴
        y0 = self.S0, self.I0, self.R0, self.V0  # 初始条件参数
        ret = scipy.integrate.odeint(self.deriv, y0, T, args=(
            self.N, self.beta, self.gamma, self.vax))
        S, I, R, V = ret.T

        # plt.plot(t, S)
        plt.figure(figsize=(8, 6))
        plt.axes(yscale="log")
        plt.plot(T, I)
        plt.plot(T, R)
        plt.plot(T, V)
        plt.plot(T, [1000]*t, linestyle="--")
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.legend(['Infect', 'Recover', 'vaccine accepted'])
        print("Final infect num:", int(I[-1]))
        print("Max infect num:", int(max(I)))
        for i in range(len(I)):
            if I[i] < 1000:
                print("End day:", i)
                break
        plt.show()

# %%


model = SIR_covid19(**parameters)
model.run(300)

# %%
