import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # row*col种状态[row*col]
        # 每个状态有4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)[[4],[4],...row*col]
        # 每个动作会进行状态转移，包含四个信息,[[[(p, next_state, reward, done)],[()],[()],[()]],[4],...row*col]
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    # max表示不会超过最小边界，min不会超过最大边界
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    
def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()# 换行

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')

            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')

            else:
                a = agent.pi[i * agent.env.ncol + j]
                # 最终行动只有一个概率为1,其余都是0
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



import numpy as np
# 状态转移矩阵，形状为(states_num,s_actions_num,next_states_num,4)  
# 其中next_states_data是一个tuple包含四个内容:(状态转移概率，下一状态，奖励值，终止与否)  
class MC_Algrithm:
    """ MC基本算法 """
    def __init__(self, env, epsilon, gamma,timestep_max,delta):
        self.diff=delta
        self.env = env
        self.epsilon=epsilon
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.timestep_max=timestep_max
        self.gamma = gamma
        # 价值迭代结束后得到的策略，初始化为均匀分布
        self.pi = [[1/len(env.P[i]) for _ in range(len(env.P[i]))] for i in range(self.env.ncol * self.env.nrow)]
        
    # 获取边界集合
    def Set_End_Disaster(self,ends,disaster):
        self.Disaster = disaster
        self.ends = ends
        
    def sample_episodes_from_SA(self,str_state,str_action, number):
        ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
        episodes = []
        for _ in range(number):
            episode = []
            timestep = 1
            s=str_state
            a=str_action
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in range(len(self.env.P[s][a])):
                temp += self.env.P[s][a][s_opt][0]
                if temp > rand:
                    s_next = self.env.P[s][a][s_opt][1]
                    r=self.env.P[s][a][s_opt][2]
                    done=self.env.P[s][a][s_opt][3]
                    break
            episode.append((s, a, r, s_next,done))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环

            while s not in self.Disaster and s not in self.ends and timestep<=self.timestep_max:
                timestep += 1
                rand, temp = np.random.rand(), 0
                # 在0-1之间生成一个随机数，这个随机数落在的区间决定选择哪个动作
                # 如A[a1,a2,a3]->P[0.2,0.3,0.5]对应的区间分别是累计概率[(0~0.2),(0.2~0.5),(0.5~1.0)]
                # 在状态s下根据策略选择动作
                for a_opt in range(len(self.pi[s])):
                    # 遍历动作集合
                    # 由于每次的遍历顺序是一致的，概率累加表示的区间也是一致的
                    temp += self.pi[s][a_opt]
                    if temp > rand:
                    # 累计概率表示的上一个动作区间不包括rand，而当前动作包括，则选择该动作
                        a = a_opt
                        break
                rand, temp = np.random.rand(), 0
                # 根据状态转移概率得到下一个状态s_next
                for s_opt in range(len(self.env.P[s][a])):
                    temp += self.env.P[s][a][s_opt][0]
                    if temp > rand:
                        s_next = self.env.P[s][a][s_opt][1]
                        r=self.env.P[s][a][s_opt][2]
                        done=self.env.P[s][a][s_opt][3]
                        break
                episode.append((s, a, r, s_next,done))  # 把（s,a,r,s_next）元组放入序列中
                s = s_next  # s_next变成当前状态,开始接下来的循环
            episodes.append(episode)
        return episodes


    # 对一个采样序列计算q_value
    def Compute_Q_SA(self,episode):
        g_sa=0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next,done) = episode[i]
            g_sa = r + self.gamma * g_sa
        return g_sa
    
    # 对所有序列计算平均q_VALUE
    def Compute_Q_mean(self,str_state,str_action, number):
        total_q_value=0
        episodes=self.sample_episodes_from_SA(str_state,str_action, number)
        for episode in episodes:
            total_q_value+=self.Compute_Q_SA(episode)
        return total_q_value/len(episodes)
    
    def MC_iteration(self,sample_num,iter_num):
        step=0
        while True:
            step+=1
            diff=0
            for state in range(len(self.pi)):
                Q_SA_means=[]
                for action in range(len(self.pi[state])):
                    Q_SA_means.append(self.Compute_Q_mean(state,action,sample_num))
                diff=max(diff,self.Policy_improve(state,Q_SA_means))
                # print(f"第{i+1}轮迭代,第{state+1}次策略提升")
            print(f"第{step}轮策略提升,策略最大误差{diff}")
            if diff<self.diff or step>iter_num: break
        self.one_hot_policy()
        print("策略优化完成")
    
    def one_hot_policy(self):
        for state in range(len(self.pi)):
            max_value = max(self.pi[state])          # 先找到最大值
            max_index = self.pi[state].index(max_value)
            self.pi[state]=[1 if i==max_index else 0 for i in range(len(self.pi[state]))]

    def Policy_improve(self,state,Q_SA_means):
        old_pi = copy.deepcopy(self.pi[state])
        old_pi=np.array(old_pi)
        maxq = max(Q_SA_means)
        cntq = Q_SA_means.count(maxq)  # 计算有几个动作得到了最大的Q值
        policy_value=0
        for a in range(len(self.pi[state])):
            if(Q_SA_means[a]==maxq):
                self.pi[state][a]=(1-self.epsilon*(len(self.pi[state])-cntq)/len(self.pi[state]))/cntq
            else:
                self.pi[state][a]=self.epsilon/len(self.pi[state])
            policy_value+=self.pi[state][a]*Q_SA_means[a]
        self.v[state]=policy_value
        new_pi=copy.deepcopy(self.pi[state])
        new_pi=np.array(self.pi[state])
        max_diff=abs(max(old_pi-new_pi))
        if state in self.Disaster or state in self.ends:
            max_diff=0
        return max_diff#返回策略概率的最大差距

env = CliffWalkingEnv()
disaster,ends=list(range(37, 47)), [47]
action_meaning = ['^', 'v', '<', '>']
epsilon = 0.0002
delta=0.01
iter_num=200
gamma = 0.9
max_step=100
sample_num=50
agent = MC_Algrithm(env, epsilon, gamma,max_step,delta)
agent.Set_End_Disaster(ends,disaster)
agent.MC_iteration(sample_num,iter_num)
print_agent(agent, action_meaning, disaster,ends)