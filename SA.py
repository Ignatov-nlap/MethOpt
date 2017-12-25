import numpy as np
import scipy as sp
import pandas as pd
from numpy import random
from numpy import matlib
import matplotlib.pyplot as plt
import math
from haversine import haversine
# import InitialPoint, Problem
from Problem import *
import warnings
from itertools import permutations
import copy
warnings.filterwarnings('ignore')


class Schedule:
    def __init__(self, problem, init_solution):
        self.problem = problem
        self.trips = []
        for i in sorted(init_solution.TripId.unique()):
            self.trips.append(Trip(np.asarray(init_solution.loc[init_solution.TripId==i].GiftId), self.problem))
#             print("Init trip {}".format(i-1))
#             self.trips[i-1].print()
        self.total_wrw = self.get_total_wrw()
        self.shift_block_length = [2, 3, 4]
        self.best_wrw = np.inf
        self.best_trips = [t.gifts.copy() for t in self.trips]
        self.moves = {
#             0:self.shift_inter,
#             1:self.shift_block_inter,
#             2:self.swap_best_inter,
#             3:self.permute_random_trip,
#             4:self.shift_random_trip,
#             5:self.swap_random_trip,
#             6:self.cut,
#             7:self.merge
            0:self.swap_random_trip,
            1:self.permute_random_trip,
            2:self.shift_random_trip,
                     }
        
    def do_random_move(self, T, verbose = False):
        if verbose: print('\n**** doing step: ****')
        move = np.random.randint(len(self.moves))
        self.moves[move](T, verbose)
        self.total_wrw = self.get_total_wrw()
        if self.total_wrw < self.best_wrw:
            self.best_wrw = self.total_wrw
            self.best_trips = [t.gifts.copy() for t in self.trips]
            
    def export_to_pandas(self):
        gift_ids = []
        trip_ids = []
        for t in range(len(self.trips)):
            gift_ids.extend(self.trips[t].gifts[1:-1]+1)
            trip_ids.extend([t+1]*(len(self.trips[t].gifts)-2))
        d = {'GiftId': gift_ids, 'TripId': trip_ids}
        return pd.DataFrame(data=d)
       
    def shift_inter(self, T, verbose=False):
        '''
        Performs shift of random gift from one random trip to another.
        '''
        # gift from t1 goes to t2
        if len(self.trips) < 2:
            return
        t_ind_1, t_ind_2 = tuple(np.random.choice(len(self.trips), 2, replace=False))
        t1 = copy.deepcopy(self.trips[t_ind_1])
        t2 = copy.deepcopy(self.trips[t_ind_2])
        weight_left = self.problem.weight_limit - t2.gifts_weight
        valid_gifts = []
        for g in range(1, len(t1.gifts)-1):
            if self.problem.weights[t1.gifts[g]] < self.problem.weight_limit - t2.gifts_weight:
                valid_gifts.append(g)
        if len(valid_gifts) == 0:
            return
        shift_ind = valid_gifts[np.random.randint(len(valid_gifts))]
        if verbose:
            print("Shifting gift {0} from trip {1} to trip {2}.".format(t1.gifts[shift_ind], t_ind_1, t_ind_2))
        delta_wrw = self.shift(t1, t2, shift_ind, verbose)
        if verbose:
            print("Delta_wrw: {0}".format(delta_wrw))
        if go_to_neighbour(delta_wrw, T):
            self.trips[t_ind_1] = t1
            self.trips[t_ind_2] = t2
            if len(self.trips[t_ind_1].gifts) <= 2:
                del self.trips[t_ind_1]
            if verbose:
                print(t1.gifts)
                print(t2.gifts)
        
    def shift(self, t1, t2, ind, verbose=False):
        '''
        Removes gift #ind from the trip #t1 and puts it to the best position in trip #t2.
        '''
        shift_gift = t1.gifts[ind]
        t1_old_wrw = t1.wrw
        t1.remove_gift(ind, problem)
        t2_old_wrw = t2.wrw
        best_pos = t2.get_best_position_for_insert(shift_gift, problem)
        t2.insert_gift(shift_gift, best_pos, problem)
        if verbose:
            print("t1_old_wrw: {0}, t1_new_wrw: {1}, \nt2_old_wrw: {2}, t2_new_wrw: {3}"
                  .format(t1_old_wrw, t1.wrw, t2_old_wrw, t2.wrw))
        delta_wrw = t1_old_wrw - t1.wrw + t2_old_wrw - t2.wrw
        return delta_wrw
        
    def shift_block_inter(self, T, verbose):
        # gifts from t1 go to t2
        if len(self.trips) < 2:
            return
        t_ind_1, t_ind_2 = tuple(np.random.choice(len(self.trips), 2, replace=False))
        t1 = copy.deepcopy(self.trips[t_ind_1])
        t2 = copy.deepcopy(self.trips[t_ind_2])
        weight_left = problem.weight_limit - t2.gifts_weight
        valid_blocks = []
        for l in self.shift_block_length:
            for first_gift in range(1, len(t1.gifts)-l):            
                block_weight = np.sum(problem.weights[t1.gifts[first_gift:first_gift+l]])
                if block_weight < problem.weight_limit - t2.gifts_weight:
                    valid_blocks.append((first_gift, l))
        if len(valid_blocks) == 0:
            return
        block_first, block_len = valid_blocks[np.random.randint(len(valid_blocks))]
        shift_gift_indices = t1.gifts[block_first:block_first+block_len]
        if verbose:
            print("Shifting block {0} from trip {1} to trip {2}.".format(shift_gift_indices, t_ind_1, t_ind_2))
        t1_old_wrw = t1.wrw
        t1.remove_block(block_first, block_len, problem)
        t2_old_wrw = t2.wrw
        best_pos = t2.get_best_position_for_block_insert(shift_gift_indices, problem)
        t2.insert_block(shift_gift_indices, best_pos, problem)
        if verbose:
            print("t1_old_wrw: {0}, t1_new_wrw: {1}, \nt2_old_wrw: {2}, t2_new_wrw: {3}"
                  .format(t1_old_wrw, t1.wrw, t2_old_wrw, t2.wrw))
        delta_wrw = t1_old_wrw - t1.wrw + t2_old_wrw - t2.wrw
        if go_to_neighbour(delta_wrw, T):
            self.trips[t_ind_1] = t1
            self.trips[t_ind_2] = t2 
            if len(self.trips[t_ind_1].gifts) <= 2:
                del self.trips[t_ind_1]

    def swap_best_inter(self, T, verbose):
        if len(self.trips) < 2:
            return
        t_ind_1, t_ind_2 = tuple(np.random.choice(len(self.trips), 2, replace=False))
        t1 = copy.deepcopy(self.trips[t_ind_1])
        t2 = copy.deepcopy(self.trips[t_ind_2])
        found = False
        for i in range(15):
            g1 = np.random.randint(1, len(t1.gifts)-1)
            g2 = np.random.randint(1, len(t2.gifts)-1)
            if t1.gifts_weight + problem.weights[t2.gifts[g2]] < self.problem.weight_limit and \
            t2.gifts_weight + problem.weights[t1.gifts[g1]] < self.problem.weight_limit:
                found = True
                break
        if not found:
            return
        delta_wrw = self.swap_gifts(t1, t2, g1, g2)
        if go_to_neighbour(delta_wrw, T):
            self.trips[t_ind_1] = t1
            self.trips[t_ind_2] = t2
        
    def swap_gifts(self, t1, t2, g1, g2):
        t1_old_wrw = t1.wrw
        t2_old_wrw = t2.wrw
        gift_id1 = t1.gifts[g1]
        gift_id2 = t2.gifts[g2]
        t1.remove_gift(g1, self.problem)
        t2.remove_gift(g2, self.problem)
        t1.insert_gift(gift_id2, t1.get_best_position_for_insert(gift_id2, problem), problem)
        t2.insert_gift(gift_id1, t2.get_best_position_for_insert(gift_id1, problem), problem)
        delta_wrw = t1_old_wrw - t1.wrw + t2_old_wrw - t2.wrw
        return delta_wrw
    
    def cut(self, T, verbose):
        t_ind = np.random.randint(len(self.trips))
        best_t1, best_t2 = None, None
        best_wrw = np.inf
        for i in range(2, len(self.trips[t_ind].gifts)-1):
            t1 = Trip(self.trips[t_ind].gifts[1:i], problem, from_pandas=False)
            t2 = Trip(self.trips[t_ind].gifts[i:-1], problem, from_pandas=False)
            if t1.wrw + t2.wrw < best_wrw:
                best_wrw = t1.wrw + t2.wrw
                best_t1, best_t2 = t1, t2
        if go_to_neighbour(self.trips[t_ind].wrw - best_wrw, T):
            del self.trips[t_ind]
            self.trips.append(best_t1)
            self.trips.append(best_t2)
            
    def merge(self, T, verbose):
        valid_merges = []
        for i in range(len(self.trips)):
            for j in range(len(self.trips)):
                if i!=j and self.trips[i].gifts_weight + self.trips[j].gifts_weight < self.problem.weight_limit:
                    valid_merges.append((i, j))
        if len(valid_merges) == 0:
            return
        m = np.random.randint(len(valid_merges))
        t1, t2 = valid_merges[m]
        t = Trip(np.concatenate([self.trips[t1].gifts[1:-1], self.trips[t2].gifts[1:-1]]),
                 problem, from_pandas=False)
        if go_to_neighbour(self.trips[t1].wrw + self.trips[t2].wrw - t.wrw, T):
#         if self.trips[t1].wrw + self.trips[t2].wrw > t.wrw:
            i, j = max(t1, t2), min(t1, t2)
            del self.trips[i]
            del self.trips[j]
            self.trips.append(t)       
#         best_merge = ()
#         best_t = None
#         best_wrw = np.inf
#         for m in range(len(valid_merges)):
#             t1, t2 = valid_merges[m]
#             t = Trip(np.concatenate([self.trips[t1].gifts[1:-1], self.trips[t2].gifts[1:-1]]), problem, from_pandas=False)
#             if t.wrw < best_wrw:
#                 best_merge = valid_merges[m]
#                 best_wrw = t.wrw
#                 best_t = t
#         if go_to_neighbour(self.trips[best_merge[0]].wrw + self.trips[best_merge[1]].wrw - best_wrw, T):
#             i, j = max(best_merge[0], best_merge[1]), min(best_merge[0], best_merge[1])
#             del self.trips[i]
#             del self.trips[j]
#             self.trips.append(best_t)
                  
    def get_total_wrw(self):
        return sum([t.wrw for t in self.trips])
    
    def permute_random_trip(self, T, verbose):
        valid_trips = []
        for t in self.trips:
            if len(t.gifts > 4):
                valid_trips.append(t)
        t_ind = np.random.randint(len(valid_trips))
        valid_trips[t_ind].permute(self.problem)
        
    def shift_random_trip(self, T, verbose):
        valid_trips = []
        for t in self.trips:
            if len(t.gifts) > 3:
                valid_trips.append(t)
        t_ind = np.random.randint(len(valid_trips))
        valid_trips[t_ind].shift(self.problem)
        
    def swap_random_trip(self, T, verbose):
        valid_trips = []
        for t in self.trips:
            if len(t.gifts) >= 3:
                valid_trips.append(t)
        t_ind = np.random.randint(len(valid_trips))
        valid_trips[t_ind].swap(self.problem, T)
        
        
def go_to_neighbour(delta_wrw, T):
    return np.exp(delta_wrw / T) > np.random.rand()

def simulated_annealing(problem, init_solution, T=8000, mu=1.00002, T_stop=1e+2, max_iter=100000+1, verbose=False):
    schedule = Schedule(problem, init_solution)
    n_iter = 0
    while T>T_stop and n_iter < max_iter:
        schedule.do_random_move(T, verbose)
        if n_iter%4000 == 0:
            if verbose: print(n_iter, T, schedule.get_total_wrw())
        if n_iter%40 == 0:
            T/=mu
        n_iter+=1
    return schedule

class Trip:
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
        