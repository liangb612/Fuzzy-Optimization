# 导入 gurobipy 库
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB, MVar

#已知变量
class staticValues :
  alpha = 0.9
  w_w = np.array([0.6,1,1.4])
  w_v = np.array([0.5,1,1.5])
  w_l = np.array([0.9,1,1.1])
  pw_f = np.array([188,237,188,181,204,156,174,186,118,89,77,54,52,80,82,107,144,185,163,221,215,240,223,190])
  pv_f =np.array([0,0,0,0,0,2.2000,5.5000,17.0000,28.6000,32.0000,39.0000,42.6000,42.0000,41.6000,40.5000,
    41.2000,36.5000,28.0000,16.0000 ,6.6000,1.1000,0,0,0])
  #pload=np.array([945,845,745 ,780,998,1095,1147,1199,1300,1397,1449,1498,1397,1297,1197,1048,1000, 1100,
   # 1202,1375,1298,1101,900,800])
  pload=np.array([945,845,745 ,780,998,1095,1147,1199,1300,1397,1449,1498,1397,1297,1197,1048,1000, 1100,
     1202,1375,1298,1101,900,800])
  p_g_min =np.array([200,200,150,120,70])
  p_g_max =np.array([460,400,350,300,150])
  p_h_min = 0
  p_h_max = 280
  remp_u_d =np.array([240,210,150,120,70])#>=pgmin
  t_on_and_off =np.array([8,7,6,4,3])
  a = 1e-4*np.array([1.02 ,1.21 ,2.17 ,3.42 ,6.63])
  b =np.array([0.277, 0.288 ,0.29 ,0.292 ,0.306])
  c =np.array([9.2, 8.8 ,7.2, 5.2 ,3.5])
  sit =np.array([25.6 ,22.3 ,16.2 ,12.3 ,4.6])
  e =np.array([0.877, 0.877 ,0.877 ,0.877, 0.979])
  lamda =np.array([0.94, 0.94, 0.94, 0.94 ,1.03])
  #储能模型中的常量
  capmax=400
  eesmax=100
  eesmin=0
  socmax=0.9
  socmin=0.2
  theta=0.01
  yita=0.95
  #碳交易
  w = 50
  d = 100
  tao = 0.25
  #循环控制变量
  # 时间
  horizon = 24
  #火力发电机组数
  n_gen = 5

class optimization_variable:
  def __init__(self):
    self.model = gp.Model()
    self.k = staticValues()
    self.p_g = self.model.addMVar((self.k.n_gen, 24), vtype=GRB.CONTINUOUS, lb=0, name="P_G")
    self.p_h = self.model.addMVar((1,24), vtype=GRB.CONTINUOUS, lb=0, name="P_H")
    self.p_b_ch =self.model.addMVar((1,24),vtype=GRB.CONTINUOUS,lb=0,name = "P_B_CH")
    self.p_b_dis =self.model.addMVar((1,24),vtype=GRB.CONTINUOUS,lb=0,name = "P_B_DIS")
    self.p_w =self.model.addMVar((1,24),vtype=GRB.CONTINUOUS,lb=0,name = "P_W")#带_的是出力优化变量，w_f的是预测值
    self.p_v =self.model.addMVar((1,24),vtype=GRB.CONTINUOUS,lb=0,name = "P_V")
    self.b_s =self.model.addMVar((1,24),vtype=GRB.BINARY,lb=0,name = "B_S")
    self.b_s2 =self.model.addMVar((1,24),vtype=GRB.BINARY,lb=0,name = "B_S2")
    self.g_s =self.model.addMVar((self.k.n_gen,24),vtype=GRB.BINARY,name = "G_S")
    self.lin = self.model.addMVar((1,24),vtype=GRB.CONTINUOUS,name = "lin")
    #分段线性化优化变量，由于model方法addGenConstrPWL的存在，无需手动进行SOS2分段线性化。
    self.p_g_2 = self.model.addMVar((self.k.n_gen, 24), vtype=GRB.CONTINUOUS, lb=0, name="P_G_2")
  #辅助优化变量：
    self.gy = self.model.addMVar((1,24),vtype=GRB.CONTINUOUS)
    self.delter_g_s =self.model.addMVar((self.k.n_gen,23),vtype=GRB.BINARY)
    self.pgsum = self.model.addMVar((1,24),vtype=GRB.CONTINUOUS)
  @staticmethod
  def getConsEES(model, x_P_ch, x_P_dis, x_u_ch, x_u_dis, EESmax, EESmin, capmax, Horizon, theta)->MVar:

      # 常量定义
      soc0 = 0.5           # 初始SOC
      yitac = 0.95         # 充电效率
      yitad = 0.95         # 放电效率
      socmax = 0.9         # SOC上限
      socmin = 0.2         # SOC下限

      # 创建SOC变量 (连续变量)
      soc = model.addMVar(Horizon, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,name="soc")
      for t in range(Horizon):
          # 下界约束
          model.addConstr(x_P_ch[0,t] >= x_u_ch[0,t] * EESmin,name=f"ch_power_lb_{t}")
          # 上界约束
          model.addConstr(x_P_ch[0,t] <= x_u_ch[0,t] * EESmax,name=f"ch_power_ub_{t}")

      # 放电功率约束: -x_u_dis * EESmax <= x_P_dis <= -x_u_dis * EESmin
      for t in range(Horizon):
        # 下界约束（放电为负）
        model.addConstr(x_P_dis[0,t] >= x_u_dis[0,t] * EESmin,name=f"dis_power_lb_{t}")
          # 上界约束
        model.addConstr(x_P_dis[0,t] <= x_u_dis[0,t] * EESmax,name=f"dis_power_ub_{t}")

      ### 2. 不同时充放电约束 ###
      # 0 <= x_u_ch + x_u_dis <= 1
      for t in range(Horizon):
          model.addConstr(x_u_ch[0,t] + x_u_dis[0,t] <= 1,
                         name=f"no_simultaneous_{t}")
          model.addConstr(x_u_ch[0,t] + x_u_dis[0,t] >= 0,
                         name=f"no_simultaneous_lb_{t}")

      model.addConstr(soc[0] == soc0 +
                      x_P_ch[0,0] * yitac / capmax -
                      x_P_dis[0,0] / yitad / capmax,
                      name="soc_eq_0")

      # 第一个时段的SOC边界
      model.addConstr(soc[0] >= socmin, name="soc_lb_0")
      model.addConstr(soc[0] <= socmax, name="soc_ub_0")

      # 后续时段的SOC约束
      for t in range(1, Horizon):
          # SOC递推关系
          model.addConstr(soc[t] == soc[t-1] * (1 - theta) +
          x_P_ch[0,t] * yitac / capmax -
          x_P_dis[0,t] / yitad / capmax,
          name=f"soc_eq_{t}")

          # SOC边界约束
          model.addConstr(soc[t] >= socmin, name=f"soc_lb_{t}")
          model.addConstr(soc[t] <= socmax, name=f"soc_ub_{t}")

      ### 4. 净能量平衡约束（周期约束）###
      # 总充放电量平衡（考虑效率）
      # 创建临时变量存储总净能量
      total_net_energy = 0
      for t in range(Horizon):
          total_net_energy += x_P_ch[0,t] * yitac / capmax - x_P_dis[0,t] / yitad / capmax

      # 加上自放电损失
      # 这是一个简化的处理，实际可能需要更精确的模型
      model.addConstr(total_net_energy == 0, name="energy_balance")
      return soc

  @staticmethod
  def add_piecewise_penalty(model, PG, e, lam, w, d, tao, Horizon):

      dd = model.addMVar((3, Horizon), vtype=GRB.BINARY, name="dd")

      lin = model.addMVar((1,Horizon),vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lin")

      for i in range(Horizon):
          # 计算当前时段的 ml 和 mp
          # PG[:,i] 是第i列，即所有机组在时段i的出力
          ml_i = gp.quicksum(e[j] * PG[j, i] for j in range(5))
          mp_i = gp.quicksum(lam[j] * PG[j, i] for j in range(5))

          # 约束1: 每时段只能选一个分段
          model.addConstr(dd[0, i] + dd[1, i] + dd[2, i] == 1,
                          name=f"select_one_{i}")

          # 大M法实现implies约束
          M = 10000  # 足够大的数，可以根据问题规模调整
          # 分段1: mp <= ml + d
          model.addConstr(mp_i <= ml_i + d + M * (1 - dd[0, i]),
                          name=f"segment1_ub_{i}")
          model.addConstr(lin[0,i] >= w * (mp_i - ml_i) - M * (1 - dd[0, i]),
                          name=f"segment1_lin_lb_{i}")
          model.addConstr(lin[0,i] <= w * (mp_i - ml_i) + M * (1 - dd[0, i]),
                          name=f"segment1_lin_ub_{i}")
          # 分段2: ml + d <= mp <= ml + 2d
          model.addConstr(mp_i >= ml_i + d - M * (1 - dd[1, i]),
                          name=f"segment2_lb_{i}")
          model.addConstr(mp_i <= ml_i + 2*d + M * (1 - dd[1, i]),
                          name=f"segment2_ub_{i}")
          model.addConstr(lin[0,i] >= (1+tao)*w*(mp_i - ml_i) - tao*w*d - M*(1 - dd[1, i]),
                          name=f"segment2_lin_lb_{i}")
          model.addConstr(lin[0,i] <= (1+tao)*w*(mp_i - ml_i) - tao*w*d + M*(1 - dd[1, i]),
                          name=f"segment2_lin_ub_{i}")
          # 分段3: mp >= ml + 2d
          model.addConstr(mp_i >= ml_i + 2*d - M * (1 - dd[2, i]),
                          name=f"segment3_lb_{i}")
          model.addConstr(lin[0,i] >= (1+2*tao)*w*(mp_i - ml_i) - 3*tao*w*d - M*(1 - dd[2, i]),
                          name=f"segment3_lin_lb_{i}")
          model.addConstr(lin[0,i] <= (1+2*tao)*w*(mp_i - ml_i) - 3*tao*w*d + M*(1 - dd[2, i]),
                          name=f"segment3_lin_ub_{i}")
      # 3. 计算总惩罚 C3 = sum(lin)
      C3 = gp.quicksum(lin[0,i] for i in range(Horizon))
      return C3

  def cons(self):
    self.model.addConstr(self.p_h<=self.k.p_h_max)
    self.model.addConstr(self.p_w<=self.k.pw_f )
    self.model.addConstr(self.p_v<=self.k.pv_f)
    # 火电约束
    #爬坡约束

    for n in range(self.k.n_gen):
        for t in range(1,self.k.horizon):
          self.model.addConstr(self.p_g[n,t].item()-self.p_g[n,t-1].item()<=self.k.remp_u_d[n])
          self.model.addConstr(self.p_g[n,t].item()-self.p_g[n,t-1].item()>=-self.k.remp_u_d[n])
          #self.model.addConstr(self.p_g[n,t].item()-self.p_g[n,t-1].item()>=1)
          '''
          self.model.addConstr(self.delter_g_s[n,t-1] == self.g_s[n,t].item()-self.g_s[n,t-1].item())
          #启动停止保持约束
          for k in range(self.k.t_on_and_off[n]):
            self.model.addConstr(self.delter_g_s[n,t-1] <= self.g_s[n,min(t+k,23)])
            self.model.addConstr(-self.delter_g_s[n,t-1] <=1- self.g_s[n,min(t+k,23)])
          '''
    for n in range(self.k.n_gen):
      for t in range(1, self.k.horizon):
                  # 如果t-1是0，t是1（开机），那么t到min(t+T_on-1, horizon-1)必须为1
        self.model.addConstr(
          self.g_s[n,t] - self.g_s[n,t-1]
          <= self.g_s[n, min(t+self.k.t_on_and_off[n]-1, self.k.horizon-1)]
        )
    for n in range(self.k.n_gen):
      for t in range(1, self.k.horizon):
        self.model.addConstr(
          self.g_s[n,t-1] - self.g_s[n,t]
          <=1- self.g_s[n, min(t+self.k.t_on_and_off[n]-1, self.k.horizon-1)]
          )

    self.model.addConstr(self.p_g<=self.g_s*np.repeat(self.k.p_g_max,24).reshape(-1,24))
    self.model.addConstr(self.p_g>=self.g_s*np.repeat(self.k.p_g_min,24).reshape(-1,24))


    # 功率平衡约束
    p_w_b =((1-self.k.alpha)*self.k.w_w[0]/2+self.k.w_w[1]/2+self.k.w_w[2]*self.k.alpha/2)*self.k.pw_f
    p_v_b =((1-self.k.alpha)*self.k.w_v[0]/2+self.k.w_v[1]/2+self.k.w_v[2]*self.k.alpha/2)*self.k.pv_f
    p_l_b =((1-self.k.alpha)*self.k.w_l[0]/2+self.k.w_l[1]/2+self.k.w_l[2]*self.k.alpha/2)*self.k.pload
    #self.gy = gp.quicksum(self.g_s[n,:].item()*self.p_g[n,:].item()#item的作用是将优化变量展开为var矩阵，以便numpy运算
      #for n in range(self.k.n_gen))
    #self.model.addConstr(self.gy == )
    for t in range(24):

      gy_temp= gp.quicksum(self.g_s[n,t].item()*self.p_g[n,t].item() for n in range(5))
      self.model.addConstr(self.gy[0,t] - gy_temp==0)#type:ignore
    print(f"清晰化pwb{p_w_b}")
    print(f"清晰化pvb{p_v_b}")
    print(f"清晰化plb{p_l_b}")
    print(f"非线性优化变量线性等效{self.gy}")
    self.model.addConstr((2-2*self.k.alpha)*(self.k.w_l[1]*self.k.pload-self.k.w_w[1]*self.p_w-self.k.w_v[2]*
      self.p_v)+(2*self.k.alpha-1)*(self.k.w_l[2]*self.k.pload-self.k.w_w[0]*self.p_w-self.k.w_v[0]*self.p_v)+self.p_b_ch-self.p_b_dis-self.p_h-self.gy == 0)#充电时电池为源与pg同号
    #旋转备用约束
    for t in range(24):
      pgsum_temp= gp.quicksum(self.g_s[n,t].item()*self.k.p_g_max[n] for n in range(self.k.n_gen))
      self.model.addConstr(self.pgsum[0,t] - pgsum_temp==0)
    print(f"Mvar变量求和{self.pgsum}")

    self.model.addConstr((2-2*self.k.alpha)*(self.k.w_l[1]*self.k.pload-self.k.w_w[1]*self.p_w-self.k.w_v[2]*
      self.p_v)+(2*self.k.alpha-1)*(self.k.w_l[2]*self.k.pload-self.k.w_w[0]*self.p_w-self.k.w_v[0]*self.p_v)+self.p_b_ch-self.p_b_dis-self.p_h-self.pgsum<= 0)

    #分段线性化
    gn=5
    gl1=(self.k.p_g_max-self.k.p_g_min)/gn
    gl2 = np.array([[self.k.p_g_min[j] + i * gl1[j] for i in range(gn + 1)] for j in range(gn)])#二维矩阵的循环创建方法
    gl3 = gl2 ** 2
    print(gl2)
    print(gl3)
    # 对每个机组、每个时段添加分段线性约束
    c1sum = 0
    not_t=self.model.addMVar((5,24),vtype=GRB.BINARY)
    self.model.addConstr(not_t<=self.g_s)
    self.model.addConstrs((not_t[n,t].item()<=1-self.g_s[n,t-1].item() for n in range(5) for t in range(1,24)))
    for n in range(self.k.n_gen):
      for t in range(self.k.horizon):
          self.model.addGenConstrPWL(self.p_g[n,t].item(), self.p_g_2[n,t].item(), gl2[n], gl3[n])

          c1sum =c1sum+ self.k.a[n]*self.p_g_2[n,t]+self.k.b[n]*self.p_g[n,t]+self.k.c[n]+self.k.sit[n]*not_t[n,t]
    self.C1=c1sum
    self.C2=gp.quicksum(500*(p_w_b[t]-self.p_w[0,t])+500*(p_v_b[t]-self.p_v[0,t])
                      for t in range(self.k.horizon))
  #碳交易成本
    self.C3 = self.add_piecewise_penalty(self.model,self.p_g, self.k.e, self.k.lamda, self.k.w, self.k.d, self.k.tao, self.k.horizon)



def main():
  c=optimization_variable()
  c.cons()
  soc = c.getConsEES(c.model,c.p_b_ch , c.p_b_dis, c.b_s,c.b_s2, c.k.eesmax, c.k.eesmin, c.k.capmax, c.k.horizon, c.k.theta)
  c.model.setObjective(c.C1+c.C2+c.C3,GRB.MINIMIZE)
  c.model.setParam('MIPGap', 0.005)
  c.model.optimize()
  PG = np.array(c.p_g.X)
  PH = np.array(c.p_h.X).flatten()
  P_ch = np.array(c.p_b_ch.X).flatten()
  P_dis =np.array(c.p_b_dis.X).flatten()
  P_w = np.array(c.p_w.X).flatten()
  P_v = np.array(c.p_v.X).flatten()
  G_s = c.g_s.X
  Gy = c.gy.X
  B_s = c.b_s.X
  Soc = soc.X
  print(P_v.shape)
  tt = np.vstack([
    PG[0,:],
    PG[1,:],  # 火电2
    PG[2,:],  # 火电3
    PG[3,:],  # 火电4
    PG[4,:],  # 火电5
    PH,
    P_ch,
    P_dis,
    P_w,
    P_v
  ])
  print(f"p_g优化结果:\n{PG}\ng_s优化结果：\n{G_s}\ngy优化结果：\n{Gy}\n")
  print(f"p_h优化结果:\n{PH}\np_ch优化结果：\n{P_ch}\nP_dis优化结果：\n{P_dis}\n")
  print(f"p_w优化结果:\n{P_w}\np_v优化结果：\n{P_v}\nb_s\n{B_s}\n{Soc}")

  # 创建堆叠柱状图
  plt.figure(figsize=(12, 6))
  bottom = np.zeros(len(tt[0]))  # 初始化底部位置
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#9364c0",
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  for i, data in enumerate(tt):
      plt.bar(range(len(data)), data, bottom=bottom,
               color=colors[i % len(colors)])
      bottom += data
      # 设置图例和标签
  plt.legend([
    "thermal generator1",
    "thermal generator2",
    "thermal generator3",
    "thermal generator4",
    "thermal generator5",
    "hydropower generator",
    "ESS charge",
    "ESS discharge",
    "wind power",
    "photovoltaic"])
  plt.xlabel("T")
  plt.ylabel("Power (MW)")
  plt.title("Contribution of the Source")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
