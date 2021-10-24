from grid2op.Agent import BaseAgent
import os
import datetime
import torch
import numpy as np
from .discrete_models import CategoricalPolicy
import math
from grid2op.Reward import  BaseReward

class Agent(BaseAgent):
    def __init__(self, env, submission_dir):

        self.env = env
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        BaseAgent.__init__(self, env.action_space)
        self.danger = 0.9
        self.thermal_limit = env._thermal_limit_a
        self.obs_list = ['p_or', 'rho', 'actual_dispatch', 'p_ex', 'prod_p', 'load_p', 'line_status',
                         'timestep_overflow']
        self.normalize = True
        self.redis_index = [22, 78, 101, 158, 168, 170, 175, 244, 260, 262, 306, 339, 344, 363, 413, 490, 498, 507, 515,
                            517, 521, 525]
        loaded_space_1 = np.load(os.path.join(submission_dir, './action_space/actions299.npy'), allow_pickle=True)
        loaded_space_2 = np.load(os.path.join(submission_dir, './action_space/WHU_action_space.npy'), allow_pickle=True)
        donotact = np.zeros((1, 519))
        ini_action_space = np.append(loaded_space_2, donotact, axis=0)
        ini_action_space[:, -25:-3] = -1
        self.agent_action_space = ini_action_space
        self.extra_agent_action_space = loaded_space_1
        self.action_size = self.agent_action_space.shape[0]
        self.extra_action_size = self.extra_agent_action_space.shape[0]
        self.last_step = datetime.datetime.now()
        self.net = CategoricalPolicy(376, 567, 512)
        self.alarm_cooldown = 0
        self.recovery_stack = []
        self.redis_cd = 0
        self.redis_array = []
        self.cd = 0
        self.recovery_threshold = 50

        self.pfentropy_size = 2
        mid = np.array(np.linspace(0.05, 0.95, 10))
        self.log_mid = np.log(1 - mid)

    def is_safe(self, obs):
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True

    def state_normalize(self, s):
        v_fix = np.nan_to_num(s)
        v_norm = np.linalg.norm(v_fix)
        if v_norm > 1e6:
            v_res = (v_fix / v_norm) * 10.0
        else:
            v_res = v_fix
        return v_res

    def convert_obs(self, obs):
        if self.normalize:
            return torch.cat(
                [torch.FloatTensor(self.state_normalize(obs._get_array_from_attr_name(i).astype(np.float32))).unsqueeze(
                    0) for i in self.obs_list], dim=-1)
        else:
            return torch.cat(
                [torch.FloatTensor(obs._get_array_from_attr_name(i).astype(np.float32)).unsqueeze(0) for i in
                 self.obs_list], dim=-1)

    def convert_act(self, a):
        action_array = self.agent_action_space[a]
        real_action = self.action_space.from_vect(action_array)

        return real_action

    def AutoLineRecCheck(self, obs):
        AutoReCflag = 0
        ReC = 0
        agent_action = self.action_space({})
        rec_reward = []
        if False in obs.line_status:
            AutoReCandidate = [Idx for Idx, i in enumerate(obs.line_status) if i == False]
            AvaLineids = np.where(obs.time_before_cooldown_line == 0)[0]
            AutoReC = list(set(AutoReCandidate).intersection(set(list(AvaLineids))))
            if len(AutoReC) > 0:
                AutoReCflag = 1
            if AutoReCflag == 1:
                for i in range(len(AutoReC)):
                    for o, e in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                        agent_action = self.action_space.reconnect_powerline(line_id=AutoReC[i], bus_or=o, bus_ex=e)
                        obs_simulate, reward_simulate, done_simulate, _ = obs.simulate(agent_action)
                        if not done_simulate:
                            ReC = 1
                            rec_reward.append([i, agent_action, obs_simulate.rho.max()])
                            break
                if ReC == 1:
                    sorted_rec_reward = sorted(rec_reward, key=lambda x: x[2])
                    agent_action = sorted_rec_reward[0][1]

        return ReC, agent_action

    def is_legal(self, action, obs):
        adict = action.as_dict()
        if 'change_bus_vect' not in adict:
            return True
        substation_to_operate = int(adict['change_bus_vect']['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:
            return False
        for line in [eval(key) for key, val in adict['change_bus_vect'][str(substation_to_operate)].items()
                     if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                return False
        return True

    def train_act(self, net, obs):
        state = self.convert_obs(obs)
        with torch.no_grad():
            action = net.clean_action(state)
        return action.detach().cpu()

    def get_region_alert(self, observation):
        zones_these_lines = set()
        zone_for_each_lines = self.env.alarms_lines_area

        lines_overloaded = [np.argmax(observation.rho)]
        for line_id in lines_overloaded:
            line_name = observation.name_line[line_id]
            for zone_name in zone_for_each_lines[line_name]:
                zones_these_lines.add(zone_name)

        zones_these_lines = list(zones_these_lines)
        zones_ids_these_lines = [self.env.alarms_area_names.index(zone) for zone in zones_these_lines]
        return zones_ids_these_lines

    def addd(self, a):
        sum = 0
        for i in a:
            if i > 1.4:
                sum += i
        return sum

    def entropycal(self, rho):
        NL = len(rho)
        R = math.log(10)
        hard_rho = 2
        Fre = np.zeros(30)
        for x in rho:
            Fre[int(x * 10)] += 1
        Fre = Fre[:10]
        Fre /= NL
        H_vec = -R * Fre * self.log_mid
        H = np.sum(H_vec)
        overflow = rho[rho >= 1]
        if len(overflow) > 0:
            H_ov_vec = -R * 1 / len(rho) * ((np.log(0.05)) + np.log(hard_rho - overflow))
            H_ov = np.sum(H_ov_vec)
            H += H_ov
        return H

    def act(self, obs, reward, done=False):
        tnow = obs.get_time_stamp()
        if self.last_step + datetime.timedelta(minutes=5) != tnow:
            self.cd = self.recovery_threshold + 1
            self.alarm_cooldown = 0
            self.recovery_stack = []
            self.redis_array = []
            self.pfentropy = list()
        self.last_step = tnow

        self.alarm_cooldown += 1
        tt = False
        act_reward = []

        RecFlag, Rec_act = self.AutoLineRecCheck(obs)
        if RecFlag:
            obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(Rec_act)
            if not done_simulate:
                return Rec_act
            else:
                p_act = Rec_act
        else:
            p_act = self.action_space({})

        obsDN_simulate, rewardDN_simulate, doneDN_simulate, infoDN_simulate = obs.simulate(p_act)
        AutoReCandidate = [Idx for Idx, i in enumerate(obs.line_status) if i == False]

        if obs.rho.max() < 0.999 and (not doneDN_simulate):
            redisp_amount = np.sum(np.abs(obs.target_dispatch))
            if redisp_amount > 0.05:
                self.redis_array = -obs.target_dispatch
            if self.is_safe(obs) and self.redis_array != [] and self.redis_cd != 1:
                p_array = np.zeros(519)
                p_array[-47:-25] = self.redis_array
                p_array[-25:-3] = -1
                try:
                    redis_act = self.action_space.from_vect(p_array) + p_act
                    obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(redis_act)
                    action_is_legal = self.env.action_space.legal_action(redis_act, self.env)
                    if not action_is_legal:
                        pass
                    elif info_simulate['is_dispatching_illegal'] or info_simulate['is_illegal']:
                        pass
                    elif self.is_safe(obs_simulate) and (not done_simulate) and (False not in obs.line_status):
                        self.redis_array = []
                        self.redis_cd = 1
                        return redis_act
                except:
                    pass

            self.cd += 1
            act = p_act

            if self.cd > self.recovery_threshold and False not in obs.line_status:
                self.cd = 0
                line_to_disconnect = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]
                actions = []
                obs_simulate = []
                reward_simulate = []
                done_simulate = []
                info_simulate = []
                act_reward = []
                for i in line_to_disconnect:
                    new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int)
                    new_line_status_array[i] = -1
                    actions.append(self.action_space({"set_line_status": new_line_status_array}))
                for action in actions:
                    obs_simulate_, reward_simulate_, done_simulate_, info_simulate_ = obs.simulate(action)
                    obs_simulate.append(obs_simulate_.rho.max())
                    reward_simulate.append(reward_simulate_)
                    done_simulate.append(done_simulate_)
                    info_simulate.append(info_simulate_)
                if True in done_simulate or np.max(obs_simulate) > 1.9:
                    for i in range(self.action_size):
                        pre_act = self.action_space.from_vect(self.agent_action_space[i])
                        obs_simulate1 = []
                        reward_simulate1 = []
                        done_simulate1 = []
                        info_simulate1 = []
                        for action in actions:
                            obs_simulate1_, reward_simulate1_, done_simulate1_, info_simulate1_ = obs.simulate(
                                p_act + pre_act + action)
                            obs_simulate1.append(obs_simulate1_.rho.max())
                            reward_simulate1.append(reward_simulate1_)
                            done_simulate1.append(done_simulate1_)
                            info_simulate1.append(info_simulate1_)
                        obs_simulate2_, reward_simulate2_, done_simulate2_, info_simulate2_ = obs.simulate(
                            p_act + pre_act)
                        if done_simulate2_:
                            continue
                        if info_simulate2_['is_dispatching_illegal'] or info_simulate2_['is_illegal']:
                            continue
                        if obs_simulate2_.rho.max() > 1.00:
                            continue
                        if True in done_simulate1:
                            continue
                        if not self.env.action_space.legal_action(p_act + pre_act, self.env):
                            continue
                        if not self.is_legal(p_act + pre_act, obs):
                            continue
                        if True in done_simulate or self.addd(obs_simulate1) < self.addd(obs_simulate):
                            if np.max(obs_simulate1) > 1.9:
                                self.cd = self.recovery_threshold + 1
                            if i not in self.redis_index:
                                act = p_act + pre_act
                                break
                            act_reward.append([p_act + pre_act, max(obs_simulate1)])
                    if len(act_reward) > 0:
                        sorted_reward = sorted(act_reward, key=lambda x: x[1])
                        act = sorted_reward[0][0]

        else:
            self.cd = self.recovery_threshold + 1
            act_reward.append([p_act, infoDN_simulate['rewards']["my_five_reward"], 566])
            top_actnum = self.action_size + 1
            encoded_act = self.train_act(self.net, obs)
            encoded_act = np.asarray(encoded_act)[0]
            sort_encoded_act = np.argsort(encoded_act)
            for i in range(1, top_actnum):
                a = sort_encoded_act[-i]
                act = self.convert_act(a) + p_act
                obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(act)

                if (obs_simulate is None) or done_simulate:
                    continue
                action_is_legal = self.env.action_space.legal_action(act, self.env)
                if not action_is_legal:
                    continue
                if info_simulate['is_dispatching_illegal'] or info_simulate['is_illegal']:
                    continue
                if not self.is_legal(act, obs):
                    continue
                if obs_simulate.rho.max() < obsDN_simulate.rho.max() or doneDN_simulate:
                    if obs_simulate.rho.max() < 0.95 and a not in self.redis_index:
                        tt = True
                        break
                    else:
                        act_reward.append([act, info_simulate['rewards']["my_five_reward"], a])
                        
            if not tt:
                sorted_reward = sorted(act_reward, key=lambda x: x[1])
                act = sorted_reward[-1][0]

                for k in range(1, len(sorted_reward)):
                    pre_act = sorted_reward[-k][0]
                    obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(pre_act)
                    if not done_simulate:
                        act = pre_act
                        break

                obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(act)

                if done_simulate or obs_simulate.rho.max() > 0.95:
                    act_reward = []
                    for i in range(self.extra_action_size):
                        pre_act = self.action_space.from_vect(self.extra_agent_action_space[i]) + p_act
                        obs_simulate_, reward_simulate_, done_simulate_, info_simulate_ = obs.simulate(pre_act)
                        if done_simulate_:
                            continue
                        action_is_legal = self.env.action_space.legal_action(act, self.env)
                        if not action_is_legal:
                            continue
                        if info_simulate['is_dispatching_illegal'] or info_simulate['is_illegal']:
                            continue
                        if not self.is_legal(act, obs):
                            continue
                        if done_simulate or obs_simulate_.rho.max() < obs_simulate.rho.max():
                            act_reward.append([pre_act, obs_simulate_.rho.max()])
                            continue

                    if len(act_reward) > 0:
                        sorted_reward = sorted(act_reward, key=lambda x: x[1])
                        act = sorted_reward[0][0]

        obs_simulate, reward_simulate, done_simulate, info_simulate = obs.simulate(act)

        current_PFEntropy = self.entropycal(obs.rho)
        self.pfentropy.append(current_PFEntropy)
        if len(self.pfentropy) > self.pfentropy_size:
            self.pfentropy.pop(0)
        current_PFE_diff = 0
        if len(self.pfentropy) == self.pfentropy_size:
            current_PFE_diff = self.pfentropy[1] - self.pfentropy[0]

        predict_PFEntropy = self.entropycal(obs_simulate.rho)
        predict_PFEntropy_diff = predict_PFEntropy - current_PFEntropy

        if (((current_PFE_diff > 0.1) and (current_PFEntropy >= 1.2)) or (
                predict_PFEntropy_diff > 0.1)) and self.alarm_cooldown > 5 and (
                False in obs.line_status and obs.time_before_cooldown_line[AutoReCandidate[0]] > 3):
            self.alarm_cooldown = 0
            zones_alert = self.get_region_alert(obs)
            act.raise_alarm = [zones_alert[0]]

        if ((current_PFEntropy >= 2) or (predict_PFEntropy >= 2)) and self.alarm_cooldown > 5 and (
                False in obs.line_status and obs.time_before_cooldown_line[AutoReCandidate[0]] > 3):
            self.alarm_cooldown = 0
            zones_alert = self.get_region_alert(obs)
            act.raise_alarm = [zones_alert[0]]

        return act

class rho_Reward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = -10
        self.reward_std = 2

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2)

other_rewards = {"my_five_reward": rho_Reward}

def make_agent(env, submission_dir):
    res = Agent(env, submission_dir)
    load_path = os.path.join(submission_dir, "./model/_episodel2rpn_icaps_2021_large_seed66_roll1_pop0_alpha0.1")
    res.net.load_state_dict(torch.load(load_path))
    res.net.eval()
    return res
