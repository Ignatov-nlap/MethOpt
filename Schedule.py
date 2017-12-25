import numpy as np
import scipy as sp
import pandas as pd
import math
import Problem, Trip
import copy

class Schedule:
    '''
    Class representing route for gifts delivering.
    
    It includes list of all trips (objects of class Trip) and contains operations leading to neighboring states.
    '''
    
    def __init__(self, problem, init_solution):
        self.problem = problem
        self.trips = []
        for i in sorted(init_solution.TripId.unique()):
            self.trips.append(Trip.Trip(np.asarray(init_solution.loc[init_solution.TripId==i].GiftId), self.problem))
#             print("Init trip {}".format(i-1))
#             self.trips[i-1].print()
        self.total_wrw = self.get_total_wrw()
        self.shift_block_length = [2, 3, 4]
        self.best_wrw = np.inf
        self.best_trips = [t.gifts.copy() for t in self.trips]
        self.moves = {
            0:self.shift_inter,
            1:self.shift_block_inter,
            2:self.swap_best_inter,
            3:self.permute_random_trip,
            4:self.shift_random_trip,
            5:self.swap_random_trip,
            6:self.cut,
            7:self.merge
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
        for t in range(len(self.best_trips)):
            gift_ids.extend(self.best_trips[t][1:-1]+1)
            trip_ids.extend([t+1]*(len(self.best_trips[t])-2))
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
        if Trip.go_to_neighbour(delta_wrw, T):
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
        t1.remove_gift(ind, self.problem)
        t2_old_wrw = t2.wrw
        best_pos = t2.get_best_position_for_insert(shift_gift, self.problem)
        t2.insert_gift(shift_gift, best_pos, self.problem)
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
        weight_left = self.problem.weight_limit - t2.gifts_weight
        valid_blocks = []
        for l in self.shift_block_length:
            for first_gift in range(1, len(t1.gifts)-l):            
                block_weight = np.sum(self.problem.weights[t1.gifts[first_gift:first_gift+l]])
                if block_weight < self.problem.weight_limit - t2.gifts_weight:
                    valid_blocks.append((first_gift, l))
        if len(valid_blocks) == 0:
            return
        block_first, block_len = valid_blocks[np.random.randint(len(valid_blocks))]
        shift_gift_indices = t1.gifts[block_first:block_first+block_len]
        if verbose:
            print("Shifting block {0} from trip {1} to trip {2}.".format(shift_gift_indices, t_ind_1, t_ind_2))
        t1_old_wrw = t1.wrw
        t1.remove_block(block_first, block_len, self.problem)
        t2_old_wrw = t2.wrw
        best_pos = t2.get_best_position_for_block_insert(shift_gift_indices, self.problem)
        t2.insert_block(shift_gift_indices, best_pos, self.problem)
        if verbose:
            print("t1_old_wrw: {0}, t1_new_wrw: {1}, \nt2_old_wrw: {2}, t2_new_wrw: {3}"
                  .format(t1_old_wrw, t1.wrw, t2_old_wrw, t2.wrw))
        delta_wrw = t1_old_wrw - t1.wrw + t2_old_wrw - t2.wrw
        if Trip.go_to_neighbour(delta_wrw, T):
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
            if t1.gifts_weight + self.problem.weights[t2.gifts[g2]] < self.problem.weight_limit and \
            t2.gifts_weight + self.problem.weights[t1.gifts[g1]] < self.problem.weight_limit:
                found = True
                break
        if not found:
            return
        delta_wrw = self.swap_gifts(t1, t2, g1, g2)
        if Trip.go_to_neighbour(delta_wrw, T):
            self.trips[t_ind_1] = t1
            self.trips[t_ind_2] = t2
        
    def swap_gifts(self, t1, t2, g1, g2):
        t1_old_wrw = t1.wrw
        t2_old_wrw = t2.wrw
        gift_id1 = t1.gifts[g1]
        gift_id2 = t2.gifts[g2]
        t1.remove_gift(g1, self.problem)
        t2.remove_gift(g2, self.problem)
        t1.insert_gift(gift_id2, t1.get_best_position_for_insert(gift_id2, self.problem), self.problem)
        t2.insert_gift(gift_id1, t2.get_best_position_for_insert(gift_id1, self.problem), self.problem)
        delta_wrw = t1_old_wrw - t1.wrw + t2_old_wrw - t2.wrw
        return delta_wrw
    
    def cut(self, T, verbose):
        t_ind = np.random.randint(len(self.trips))
        best_t1, best_t2 = None, None
        best_wrw = np.inf
        for i in range(2, len(self.trips[t_ind].gifts)-1):
            t1 = Trip.Trip(self.trips[t_ind].gifts[1:i], self.problem, from_pandas=False)
            t2 = Trip.Trip(self.trips[t_ind].gifts[i:-1], self.problem, from_pandas=False)
            if t1.wrw + t2.wrw < best_wrw:
                best_wrw = t1.wrw + t2.wrw
                best_t1, best_t2 = t1, t2
        if Trip.go_to_neighbour(self.trips[t_ind].wrw - best_wrw, T):
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
        t = Trip.Trip(np.concatenate([self.trips[t1].gifts[1:-1], self.trips[t2].gifts[1:-1]]),
                 self.problem, from_pandas=False)
        if Trip.go_to_neighbour(self.trips[t1].wrw + self.trips[t2].wrw - t.wrw, T):
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
#             t = Trip(np.concatenate([self.trips[t1].gifts[1:-1], self.trips[t2].gifts[1:-1]]), self.problem, from_pandas=False)
#             if t.wrw < best_wrw:
#                 best_merge = valid_merges[m]
#                 best_wrw = t.wrw
#                 best_t = t
#         if Trip.go_to_neighbour(self.trips[best_merge[0]].wrw + self.trips[best_merge[1]].wrw - best_wrw, T):
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
            if len(t.gifts) >= 4:
                valid_trips.append(t)
        t_ind = np.random.randint(len(valid_trips))
        valid_trips[t_ind].swap(self.problem, T)
        