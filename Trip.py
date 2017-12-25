import numpy as np
import scipy as sp
import pandas as pd
import math
import Problem, Trip
import copy
from itertools import permutations

def go_to_neighbour(delta_wrw, T):
    return np.exp(delta_wrw / T) > np.random.rand()

class Trip:
    '''
    Class representing order of delivering gifts in one trip.
    '''
    
    def __init__(self, gifts, problem, from_pandas=True):
        if from_pandas:
            gifts = np.array(gifts)-1
        self.gifts = np.insert(gifts, [0, len(gifts)], -1)
        self.cum_sum = np.zeros(len(self.gifts))
        self.cum_sum[-1] = problem.weight_sleighs
        for i in range(len(self.gifts)-2, 0, -1):
            self.cum_sum[i] = self.cum_sum[i+1] + problem.weights[self.gifts[i]]
        self.gifts_weight = self.cum_sum[1] - problem.weight_sleighs
        self.update_wrw(problem)
        
    def update_wrw(self, problem):
        """
        Updates weighted reindeer weariness of the trip.
        """
        self.wrw = 0
        for i in range(1, len(self.gifts)):
            self.wrw += problem.dist[self.gifts[i-1], self.gifts[i]] * self.cum_sum[i]
#         self.check_wrw_validity(problem)
        
    def check_wrw_validity(self, problem):
        wrw = problem.weight_sleighs*problem.dist[-1, self.gifts[-2]]
        w = problem.weight_sleighs
        if not math.isclose(self.cum_sum[-1], w, rel_tol=1e-06):
            raise BaseException("Cum sums doesn't match! w={0}, self.cum_sum[-1]={1}".format(w, self.cum_sum[-1]))
        for i in range(len(self.gifts)-2, 0, -1):
            w+=problem.weights[self.gifts[i]]
            if not math.isclose(self.cum_sum[i], w, rel_tol=1e-06):
                raise BaseException("Cum sums doesn't match! w={0}, self.cum_sum[{1}]={2}".format(w, i, self.cum_sum[i]))
            wrw+= problem.dist[self.gifts[i], self.gifts[i-1]]*w
        if not math.isclose(wrw, self.wrw, rel_tol=1e-06):
            raise BaseException('WRWs does not match! wrw={0}, self.wrw={1}'.format(wrw, self.wrw))
            
    def print(self):
        print(self.gifts)
        print(self.cum_sum)
            
    def insert_gift(self, gift_id, ind, problem):
        """
        Inserts gift to position @ind in a trip.
        """
        if problem.weights[gift_id] > problem.weight_limit - self.gifts_weight:
            raise ValueError("Gift can not be put into the trip route due to the weight limit!")
        self.gifts = np.insert(self.gifts, ind, gift_id)
        # print(self.gifts)
        for i in range(1, ind):
            self.cum_sum[i] += problem.weights[gift_id]
        self.cum_sum = np.insert(self.cum_sum, ind, self.cum_sum[ind] + problem.weights[gift_id])
        self.gifts_weight += problem.weights[gift_id]
        self.update_wrw(problem)
        
    def insert_block(self, gift_ids, ind, problem):
        """
        Inserts block of gifts to position @ind in a trip.
        """
        block_weight = np.sum(problem.weights[gift_ids])
        if block_weight > problem.weight_limit - self.gifts_weight:
            raise ValueError("Block can not be put into the trip route due to the weight limit!")
        self.gifts = np.insert(self.gifts, ind, gift_ids)
        for i in range(1, ind):
            self.cum_sum[i] += block_weight
        self.cum_sum = np.insert(self.cum_sum, ind, np.zeros(len(gift_ids)))
        self.gifts_weight += block_weight
        for i in range(ind+len(gift_ids)-1, ind-1, -1):
            self.cum_sum[i] = self.cum_sum[i+1] + problem.weights[self.gifts[i]]
        self.update_wrw(problem)
        
    def get_best_position_for_block_insert(self, gift_ids, problem):
        """
        Calculates best position in a trip to insert block of gifts to.
        """
        w = np.sum(problem.weights[gift_ids])
        if w > problem.weight_limit - self.gifts_weight:
            raise ValueError("Block can not be put into the trip route due to the weight limit!")
        block_dist = 0
        block_wrw = 0
        w_left = w
        for i in range(len(gift_ids)-1):
            block_dist += problem.dist[gift_ids[i], gift_ids[i+1]]
            block_wrw += problem.dist[gift_ids[i], gift_ids[i+1]] * w_left
            w_left -= problem.weights[gift_ids[i]]
            
        best_ind = 1
        prev_wrw = self.wrw - problem.dist[self.gifts[0], self.gifts[1]] * self.cum_sum[1] + \
                              problem.dist[self.gifts[0], gift_ids[0]] * (self.cum_sum[1] + w) + \
                              block_dist * self.cum_sum[1] + block_wrw + \
                              problem.dist[gift_ids[-1], self.gifts[1]] * self.cum_sum[1]
        best_wrw = prev_wrw
        for i in range(1, len(self.gifts)-2):
            prev_wrw = prev_wrw - \
                        block_dist * problem.weights[self.gifts[i+1]] - \
                        problem.dist[self.gifts[i], gift_ids[0]] * (self.cum_sum[i+1] + w) - \
                        problem.dist[gift_ids[-1], self.gifts[i+1]] * self.cum_sum[i+1] - \
                        problem.dist[self.gifts[i+1], self.gifts[i+2]] * self.cum_sum[i+2] + \
                        problem.dist[self.gifts[i], self.gifts[i+1]] * (self.cum_sum[i+1] + w) + \
                        problem.dist[self.gifts[i+1], gift_ids[0]] * (self.cum_sum[i+2] + w) + \
                        problem.dist[gift_ids[-1], self.gifts[i+2]] * self.cum_sum[i+2]
            if prev_wrw < best_wrw:
                best_ind = i+1
        return best_ind
    
    def get_best_position_for_insert(self, gift_id, problem):
        """
        Calculates best position in a trip to insert gift to.
        """
        return self.get_best_position_for_block_insert([gift_id], problem)  
        
    def remove_gift(self, ind, problem):
        """
        Removes a gift from position @ind.
        """
        gift_weight = problem.weights[self.gifts[ind]]
        self.gifts = np.delete(self.gifts, ind)
        for i in range(1, ind):
            self.cum_sum[i] -= gift_weight
        self.cum_sum = np.delete(self.cum_sum, ind)
        self.gifts_weight -= gift_weight
        self.update_wrw(problem)
        
    def remove_block(self, ind, length, problem):
        """
        Removes a block of gifts from position @ind.
        """
        indices = np.arange(ind, ind+length)
        block_weight = np.sum(problem.weights[self.gifts[ind:ind+length]])
        self.gifts = np.delete(self.gifts, indices)
        for i in range(1, ind):
            self.cum_sum[i] -= block_weight
        self.cum_sum = np.delete(self.cum_sum, indices)
        self.gifts_weight -= block_weight
        self.update_wrw(problem)
        
    def permute(self, problem, N=5, verbose=False):
        """
        Permutes N consequent elements at a random location and keeps the best one.
        """
        N = min(N, len(self.gifts)-2)
        i = np.random.randint(1, len(self.gifts)-N)
        best_permutation = tuple(np.arange(i, i+N))
        best_wrw = prev_range_wrw= 1e+9
        if_first=True
        for p in permutations(range(i, i+N)):
            cur_wrw = problem.dist[self.gifts[i-1], self.gifts[p[0]]] * self.cum_sum[i]
            cur_weight = self.cum_sum[i] - problem.weights[self.gifts[p[0]]]
            for ind in range(N-1):
                cur_wrw += problem.dist[self.gifts[p[ind]], self.gifts[p[ind+1]]] * cur_weight
                cur_weight -= problem.weights[self.gifts[p[ind+1]]]
            cur_wrw += problem.dist[self.gifts[p[-1]], self.gifts[i+N]] * self.cum_sum[i+N]
            if cur_wrw < best_wrw:
                best_wrw = cur_wrw
                best_permutation = p
            if if_first:
                prev_range_wrw = cur_wrw
                if_first=False
        if verbose:
            print(best_permutation)
        best_permutation = np.asarray(best_permutation)
        if best_wrw<prev_range_wrw:
            if verbose:
                print(self.gifts[i:i+N])
                print(self.gifts[best_permutation])
            self.gifts[i:i+N] = self.gifts[best_permutation]
            for ind in range(i+N-1, i-1, -1):
                self.cum_sum[ind] = self.cum_sum[ind+1] + problem.weights[self.gifts[ind]]
            #self.wrw = self.wrw - prev_range_wrw + best_wrw
            self.update_wrw(problem)
        
    def shift(self, problem):
        """
        Changes position of a random gift in a trip.
        """
        i, j = tuple(np.random.choice(np.arange(1, len(self.gifts)-1), 2, replace=False))
        new_wrw = self.wrw
        if j > i:
            new_wrw = new_wrw - problem.dist[self.gifts[i-1], self.gifts[i]] * self.cum_sum[i] - \
                                problem.dist[self.gifts[i], self.gifts[i+1]] * self.cum_sum[i+1] + \
                                problem.dist[self.gifts[i-1], self.gifts[i+1]] * self.cum_sum[i]
            for ind in range(i+1, j):
                new_wrw += problem.dist[self.gifts[ind], self.gifts[ind+1]] * problem.weights[self.gifts[i]]
            new_wrw = new_wrw - problem.dist[self.gifts[j], self.gifts[j+1]] * self.cum_sum[j+1] + \
                                problem.dist[self.gifts[j], self.gifts[i]] * \
                                (self.cum_sum[j+1] + problem.weights[self.gifts[i]]) + \
                                problem.dist[self.gifts[i], self.gifts[j+1]] * self.cum_sum[j+1]
        else:
            new_wrw = new_wrw - problem.dist[self.gifts[j-1], self.gifts[j]] * self.cum_sum[j] + \
                                problem.dist[self.gifts[j-1], self.gifts[i]] * self.cum_sum[j] + \
                                problem.dist[self.gifts[i], self.gifts[j]] * \
                                (self.cum_sum[j] - problem.weights[self.gifts[i]])

            for ind in range(j, i-1):
                new_wrw -= problem.dist[self.gifts[ind], self.gifts[ind+1]] * problem.weights[self.gifts[i]]
            new_wrw = new_wrw - problem.dist[self.gifts[i-1], self.gifts[i]] * self.cum_sum[i] - \
                                problem.dist[self.gifts[i], self.gifts[i+1]] * self.cum_sum[i+1] + \
                                problem.dist[self.gifts[i-1], self.gifts[i+1]] * self.cum_sum[i+1]
        if new_wrw < self.wrw:
            if j > i:
                gid = self.gifts[i]
                self.gifts = np.delete(self.gifts, i)
                self.gifts = np.insert(self.gifts, j, gid)
                for ind in range(j, i-1, -1):
                    self.cum_sum[ind] = self.cum_sum[ind+1] + problem.weights[self.gifts[ind]]
            else:
                self.gifts = np.insert(self.gifts, j, self.gifts[i])
                self.gifts = np.delete(self.gifts, i+1)
                for ind in range(j+1, i+1):
                    self.cum_sum[ind] = self.cum_sum[ind-1] - problem.weights[self.gifts[ind-1]]
            self.wrw = new_wrw
            self.check_wrw_validity(problem)
        
    def swap(self, problem, T):
        '''
        Swaps random gift with the next one.
        '''
        i = np.random.randint(1, len(self.gifts)-2)
        w1 = problem.weights[self.gifts[i]]
        w2 = problem.weights[self.gifts[i+1]]
        new_wrw = self.wrw - \
            problem.dist[self.gifts[i-1], self.gifts[i]] * self.cum_sum[i] - \
            problem.dist[self.gifts[i+1], self.gifts[i+2]] * self.cum_sum[i+2] + \
            problem.dist[self.gifts[i-1], self.gifts[i+1]] * self.cum_sum[i] + \
            problem.dist[self.gifts[i], self.gifts[i+1]] * (w1-w2) + \
            problem.dist[self.gifts[i], self.gifts[i+2]] * self.cum_sum[i+2]
        if go_to_neighbour(self.wrw - new_wrw, T):
            self.cum_sum[i+1] = self.cum_sum[i] - problem.weights[self.gifts[i+1]]
            self.gifts[i], self.gifts[i+1] = self.gifts[i+1], self.gifts[i]
            #self.wrw = new_wrw
            self.update_wrw(problem)

