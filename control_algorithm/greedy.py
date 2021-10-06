import numpy as np

"""
    This class contains some MAB's algorithms.

    @author Alison Carrera

"""

class EpsilonGreedy(object):
    def __init__(self, n_arms, action, eps=20, gamma=0.2):
        """
            EpsilonGreedy constructor.

            :param n_arms: Number of arms which this instance need to perform.
        """
        self.data_list = list()
        self.n_arms = n_arms
        self.number_of_selections = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        self.action = action

        self.gamma = gamma
        self.eps = eps

    def select(self):
        print("reward", np.round(self.rewards, 4))
        p_arms = np.exp(self.rewards * self.eps) / np.sum(np.exp(self.rewards * self.eps))
        print("p of arms ", np.round(p_arms, 2))
        chosen_arm = np.random.choice(list(range(self.n_arms)), p=p_arms)
        

        print("chosen arm", chosen_arm)

        return chosen_arm

    def add_data(self, arm, rwd):
        self.data_list.append((arm, rwd))
        
        self.number_of_selections = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)

        for a, r in self.data_list[-50:]:
            self.reward(a, r)

    def reward(self, arm, rwd):
        """
            This method gives a reward for a given arm.

            :param chosen_arm: Value returned from select().
        """
        self.number_of_selections[arm] += 1
        if self.number_of_selections[arm] == 1:
            self.rewards[arm] = rwd
        else:
            self.rewards[arm] += self.gamma * (rwd - self.rewards[arm])


class EpsilonGreedyCost(object):
    def __init__(self, n_arms, action, eps=20, gamma=0.2):
        """
            EpsilonGreedy constructor.

            :param n_arms: Number of arms which this instance need to perform.
        """
        self.data_list = list()
        self.n_arms = n_arms
        self.number_of_selections = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        self.rewards[:5] = 0.1
        self.action = action

        self.gamma = gamma
        self.eps = eps

    def select(self):
        print("reward", np.round(self.rewards, 4))
        p_arms = np.exp(self.rewards * self.eps) / np.sum(np.exp(self.rewards * self.eps))
        print("p of arms ", np.round(p_arms, 2))
        chosen_arm = np.random.choice(list(range(self.n_arms)), p=p_arms)
        

        print("chosen arm", chosen_arm)

        return chosen_arm

    # def add_data(self, arm, rwd):
    #     rwd_bk = rwd
    #     if rwd < 0:
    #         positive_num = 0
    #         positive_rwd = 0
    #         for past_rwd in self.rewards:
    #             if past_rwd > 0:
    #                 positive_num += 1
    #                 positive_rwd += past_rwd
    #         if positive_num > 0:
    #             rwd = - positive_rwd / positive_num * self.action[arm]
    #         else:
    #             rwd = 0.0
    #     print("Client fraction: {}, reward: {}, rectified rwd: {}".format(self.action[arm], np.round(rwd_bk, 4), np.round(rwd, 4)))
    #     self.data_list.append((arm, rwd))
    #     self.number_of_selections = np.zeros(self.n_arms)
    #     self.rewards = np.zeros(self.n_arms)

    #     for a, r in self.data_list[-50:]:
    #         self.reward(a, r)

    def add_data(self, arm, rwd):
        rwd_bk = rwd
        if rwd <= 0:
            rwd = rwd * self.action[arm]
        else:
            rwd = rwd / self.action[arm]
        print("Client fraction: {}, reward: {}, rectified rwd: {}".format(self.action[arm], np.round(rwd_bk, 4), np.round(rwd, 4)))
        self.data_list.append((arm, rwd))
        self.number_of_selections = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)

        for a, r in self.data_list[-100:]:
            self.reward(a, r)

    def reward(self, arm, rwd):
        """
            This method gives a reward for a given arm.

            :param chosen_arm: Value returned from select().
        """
        self.number_of_selections[arm] += 1
        # if self.number_of_selections[arm] == 1:
        #     self.rewards[arm] = rwd
        # else:
        self.rewards[arm] += self.gamma * (rwd - self.rewards[arm])
