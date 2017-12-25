import pandas as pd;
import numpy as np;
from haversine import haversine
from gurobipy import *
import gurobipy as grb
from gurobipy import GRB
import cvxpy as cvx
from numpy import random
from SA import *

class Problem:
    
    def __init__(self, data, sample_submission=None):
        
        """
        input:
        
        data - данные об исходной проблеме в формате dataframe pandas со столбцами 
        GiftId, Latitude, Longitude, Weight
        
        sample_submission - начальное решение проблемы. Формат dataframe pandas со столбцами GiftId and TripId.
        GiftId should be ordered by the order of delivery, and different trips should have different TripIds. 
        
        """
        
        self.df_data=data
        # количество точек, включая полюс 
        self.N = data.shape[0]+1
        # вес саней
        self.weight_sleighs=10
        # лимит саней
        self.weight_limit = 1000 
        # вес подарков плюс сани
        self.weights=np.asarray(data.Weight)
        # self.weights=np.append(self.weights, self.weight_sleighs)
    
        # стартовая позиция (Lat=90, Long=0)
        # широта, долгота
        self.lat=np.asarray(data.Latitude)*(np.pi/180.0)
        self.lat=np.append(self.lat, 90*np.pi/180)
        self.long=np.asarray(data.Longitude)*(np.pi/180.0)
        self.long=np.append(self.long,0)
        self.Earth_radius=6371
        
        # разница в широте
        dlat = \
            np.matlib.repmat(self.lat,self.N,1) - \
            np.matlib.repmat(self.lat,self.N,1).transpose();
        # разница в долготе
        dlong = \
            np.matlib.repmat(self.long,self.N,1) - \
            np.matlib.repmat(self.long,self.N,1).transpose();
        
        # cos широты
        cos_lat=np.cos(self.lat).reshape(1,self.N)
        
        # harvesine distance
        # https://en.wikipedia.org/wiki/Great-circle_distance
        # расстояние между точками
        self.dist=self.Earth_radius*2* \
                 np.arcsin(np.sqrt(np.sin(dlat/2.0)**2 \
                 +cos_lat*cos_lat.T*np.sin(dlong/2.0)**2))  
    
        # решение
        self.solution=sample_submission
    
        # значение целевой функции
        self.obj=np.inf
        # weighted_reindeer_weariness(self.solution)

    def weighted_onetrip_weariness(self, onetrip): 

        """
        input:

        onetrip - последовательность развоза подарков для одного трипа (с северного полюса и обратно). 
        формат numpy array в которых содержатся GiftId.GiftId should be ordered by the order of delivery. 

        output: 

        obj - значение целевой функции для одного трипа (формат float)

        """

        obj_onetrip = 0.0          
        prev_stop = self.N-1 #начинаем с северного полюса, идем в обратном направлении по маршруту
        prev_weight = self.weight_sleighs
        for i in onetrip[::-1]:
            obj_onetrip += self.dist[prev_stop,i] * prev_weight
            prev_stop = i
            prev_weight += self.weights[i]
            
        if prev_weight > self.weight_limit + self.weight_sleighs:
            return np.inf

        # плюс маршрут с северного полюса    
        obj_onetrip += self.dist[prev_stop,self.N-1] * prev_weight

        return obj_onetrip
        

    def objective(self, all_trips):
        
        """
        input:
        
        all_trips - решение исходной проблемы в формате dataframe pandas со столбцами GiftId and TripId.
        GiftId should be ordered by the order of delivery, and different trips should have different TripIds. 
        
        output: 
        
        obj - значение целевой функции (формат float)
        
        """
        
        n=all_trips.shape[0]
        
        uniq_gifts = all_trips.GiftId.unique()
    
        uniq_trips = all_trips.TripId.unique()

        obj = 0.0
        for i in uniq_trips:
            trip = np.asarray(all_trips[all_trips.TripId==i].GiftId)-1
            wrw = self.weighted_onetrip_weariness(trip)
            if wrw == np.inf:
                print("One of the sleighs is over the weight limit!")
                return np.inf
            # print('wrw',i, ' ', wrw)
            obj += wrw
         
        if n!=len(uniq_gifts):
            return np.inf

        return obj/100000
    
    def solve_knapsack(self, y, n_sol_to_collect=100, gap=0.5, n_var=5, verbose=False, pr=False):
    
        """
        input:

        y - текущее решение дуальной задачи в формате np array длины N -кол-во подарков
        
        n_sol_to_collect - максимальное число лучших решений задачи о рюкзаке, которое пойдет в output (int)
        
        gap - Limit the search space by setting a gap for the worst possible solution that will be accepted (float)
        
        n_var - ограничение на количество переменных равных 1 в задаче о рюкзаке (int)
        
        verbose - вывод информации о решении gurobi (boolean)
        

        output: 

        alpha - массив K лучших целочисленных решений задачи в формате boolean np array N x K, 
        N - число подарков, K - число лучших решений

        """

        try:
            # Sample data
            Groundset    = range(self.N-1)
            objCoef      = y
            knapsackCoef = self.weights
            Budget       = self.weight_limit-self.weight_sleighs

            # Create initial model
            model = Model("poolsearch")
            model.setParam(GRB.Param.OutputFlag, verbose)

            # Create dicts for tupledict.prod() function
            objCoefDict = dict(zip(Groundset, objCoef))
            knapsackCoefDict = dict(zip(Groundset, knapsackCoef))

            # Initialize decision variables for ground set:
            # x[e] == 1 if element e is chosen
            Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')


            # Set objective function
            model.ModelSense = GRB.MAXIMIZE
            model.setObjective(Elem.prod(objCoefDict))

            # Constraint: limit total number of elements to be picked to be at most
            # Budget
            model.addConstr(Elem.prod(knapsackCoefDict) <= Budget, name='Budget')

            model.addConstr(grb.quicksum(Elem) <= n_var, name='nonzero var')


            # Limit how many solutions to collect
            model.setParam(GRB.Param.PoolSolutions, n_sol_to_collect)
            # Limit the search space by setting a gap for the worst possible solution that will be accepted
            model.setParam(GRB.Param.PoolGap, gap)
            # do a systematic search for the k-best solutions
            model.setParam(GRB.Param.PoolSearchMode, 2)

            # save problem
            # model.write('poolsearch.lp')

            # Optimize
            model.optimize()


            # Status checking
            status = model.Status
            if status == GRB.Status.INF_OR_UNBD or \
               status == GRB.Status.INFEASIBLE  or status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is infeasible or unbounded')
                sys.exit(1)

            if status != GRB.Status.OPTIMAL:
                print('Optimization was stopped with status ' + str(status))
                sys.exit(1)

            # Print number of solutions stored
            nSolutions = model.SolCount
            if pr: print('Number of solutions found: ' + str(nSolutions))
                
            # Print objective values of solutions
            model.setParam(GRB.Param.SolutionNumber, 0)
            s1=model.PoolObjVal/100000
            model.setParam(GRB.Param.SolutionNumber, nSolutions-1)
            s2=model.PoolObjVal/100000
            # print('s1', s1,'s2', s2, 'gap*s1', gap*s1)
                
            #for e in range(nSolutions):
                #model.setParam(GRB.Param.SolutionNumber, e)
                #print('%g ' % model.PoolObjVal, end='')
                #if e % 15 == 14:
                #     print('')
                #print('')

            alpha=[]
            for i in range(nSolutions):
                model.setParam(GRB.Param.SolutionNumber, i)
                alpha.append([True if Elem[e].Xn > .9 else False for e in Groundset])

            return np.array(alpha)


        except GurobiError as e:
            print('Gurobi error ' + str(e.errno) + ": " + str(e.message))

        except AttributeError as e:
            print('Encountered an attribute error: ' + str(e))
            
            
    def solve_dual(self, n_sol_to_collect=100, gap=0.5, pr=False, SA=True):
        
        """
        input:
        
        n_sol_to_collect - максимальное число лучших решений задачи о рюкзаке, которое пойдет в output (int)
        
        gap - Limit the search space by setting a gap for the worst possible solution that will be accepted (float)
            

        output: 

        A - транспанированная матрица ограничений для первичной задачи (np array)
        
        wtw -  лист коэффициентов для целевой функции первичной задачи (np array)
        
        wtw_solution_list - Cоответствуют коэффициентам wtw. Решения - последовательность развоза подарков для одного трипа, 
        в которых содержатся GiftId. GiftId should be ordered by the order of delivery.
        
      
        """
    
        
        if pr: print('dual problem')
        
        
        # начальные ограничения - для каждого подарка оцениваем трип в котором доставлеям только этот подарок - 
        # расстояние до подарка на вес и от подарка на вес саней
        
        wtw=[]
        wtw_solution_list=[]

        for i in range(self.N-1):
            wtw.append(self.dist[i, self.N-1]*(self.weights[i]+2*self.weight_sleighs))
            wtw_solution_list.append(np.array([i]))

        wtw=np.array(wtw)
        if pr: print('initial wtw', wtw)
        if pr: print('wtw_solution_list', wtw_solution_list)

        # Create two scalar optimization variables.
        y = cvx.Variable(self.N-1)

        A = np.eye(self.N-1, dtype=int)
        b = wtw

        # Create constraints.
        constraints = [A*y <= b,
                       y >= 0]

        # Form objective.
        obj = cvx.Maximize(sum(y))

        T=True
        iteration=0
        while T: 
            iteration+=1
            T=False

            # Form and solve problem.
            prob = cvx.Problem(obj, constraints)
            prob.solve()  # Returns the optimal value.

            # получаем решение дуальной проблемы
            y_solution=np.array(y.value).flatten()
            if pr: print('y_solution', y_solution)

            # находим трип, который предположительно будет нарушать ограничения дуальной задачи
            for n_var in range(self.N-1, 2,-1):

                if pr: print('n_var', n_var)
                alpha=self.solve_knapsack(y_solution, n_sol_to_collect=n_sol_to_collect, gap=gap, n_var=n_var)
                if pr: print(alpha)

                for a in alpha:

                    # получаем правую часть неравенства дуальной задачи для нашей альфа
                    if pr: print('a', a)
                    if pr: print('sum_a', sum(a))
                    if sum(a)<2: break
                               
                    if SA: wtw_solution=self.solve_one_trip_sa(a)
                    else: wtw_solution=self.solve_onetrip_greedy(a)
                        
                    if pr: print('wtw_solution', wtw_solution)
                    wtw_objective=self.weighted_onetrip_weariness(wtw_solution)

                    if pr: print('wtw_objective', wtw_objective)

                    if a@y_solution>wtw_objective+0.1:
                        if pr: print('добавляем a', a)
                        if pr: print('y_solution', y_solution)
                        if pr: print('добавляем ограничение, разница = ', a@y_solution-wtw_objective)
                        A=np.append(A,[a], axis=0)
                        if pr: print(A)
                        wtw=np.append(wtw,wtw_objective)
                        wtw_solution_list.append(wtw_solution)
                        if pr: print('wtw_objective', wtw_objective)
                        if pr: print('wtw_solution_list', wtw_solution_list)

                        constraints.append(a*y <=wtw_objective)
                        T=True
                        # break
                        
                        
                # возможно не надо
                # if T: break

        if pr: print(A, wtw, wtw_solution_list)
        
        if pr: print('A', A)
        if pr: print('iteration ', iteration)
        if pr: print("status:", prob.status) 
        if pr: print("optimal value", prob.value) 
        if pr: print("optimal var", y_solution) 

        return A, wtw, wtw_solution_list
    
    def solve_prime(self, n_sol_to_collect=100, gap=0.5, pr=False, verbose=False, SA=True):
       
        """
        input:
        
        n_sol_to_collect - максимальное число лучших решений задачи о рюкзаке, которое пойдет в output (int)
        
        gap - Limit the search space by setting a gap for the worst possible solution that will be accepted (float)
      
        """
    
        
        A, wtw, wtw_solution_list = self.solve_dual(n_sol_to_collect=n_sol_to_collect, gap=gap, SA=SA)

        if pr: print('wtw', wtw)

        if pr: print('\nprime problem')

        n=len(wtw)

        m = grb.Model()

        x = []
        for i_f in range(n):
            x.append(m.addVar(vtype=GRB.BINARY))

        for i_c in range(A.shape[1]):  
            constr_summands = [ -A[i_r][i_c]*x[i_r] for i_r in range(A.shape[0])]
            m.addConstr(sum(constr_summands), GRB.LESS_EQUAL, -1)

        m.modelSense = GRB.MINIMIZE

        m.update()

        obj_summands = []
        for i in range(n): 
            obj_summands.append(wtw[i]*x[i])

        m.setObjective(grb.quicksum(obj_summands))

        m.setParam(GRB.Param.OutputFlag, verbose)
        m.optimize()
        if pr: print('Status', m.Status)
       
        x_solution=np.array([x[e].Xn for e in range(n)])

        if pr: print('x_solution', x_solution)

        if pr: print(wtw_solution_list)
            
        # формируем решение в форме submission
        gifts=np.arange(1, self.N)
        TripId=0
        d = {'GiftId': [], 'TripId': []}
        for i_x, sol in enumerate(x_solution):
            if sol==1:
                d['GiftId'].extend(wtw_solution_list[i_x]+1)
                d['TripId'].extend([TripId]*len(wtw_solution_list[i_x]))
                if pr: print('GiftId',d['GiftId'])
                if pr: print('TripId', d['TripId'])
                TripId+=1

        solution = pd.DataFrame(data=d)

        self.solution=solution

        uniq_gifts = self.solution.GiftId.unique()

        if (self.N-1)==len(uniq_gifts):
            self.obj=self.objective(self.solution)
        else: self.obj=np.inf
            
        return solution
 
    def solve_onetrip_greedy(self, knapsack, pr=False):

        """
        input:

        knapsack - целочисленное решение задачи о рюкзаке в формате boolean np array

        output: 

        solution - решение задачи развоза подарков для одного трипа в формате boolean np array - 
        в которых содержатся GiftId. GiftId should be ordered by the order of delivery. 

        """

        if pr: print('knapsack', knapsack)
        index=np.where(knapsack==True)[0]
        if pr: print('subproblem index', index)
        if pr: print('length subproblen', len(index))
        subproblem=Problem(self.df_data.iloc[knapsack], None)

        df_solution=subproblem.Greedy_solver()
        subproblem_solution=np.array(df_solution['GiftId'])-1
        if pr: print('subproblem_solution', subproblem_solution)

        solution=index[subproblem_solution]
        

        return solution
    
    def solve_onetrip_greedy_deep(self, knapsack, pr=False):

        """
        input:

        knapsack - целочисленное решение задачи о рюкзаке в формате boolean np array

        output: 

        solution - решение задачи развоза подарков для одного трипа в формате boolean np array - 
        в которых содержатся GiftId. GiftId should be ordered by the order of delivery. 

        """

        if pr: print('knapsack', knapsack)
        index=np.where(knapsack==True)[0]
        if pr: print('subproblem index', index)
        if pr: print('length subproblen', len(index))
        subproblem=Problem(self.df_data.iloc[knapsack], None)

        df_solution=subproblem.Greedy_solver_deep()
        subproblem_solution=np.array(df_solution['GiftId'])-1
        if pr: print('subproblem_solution', subproblem_solution)

        solution=index[subproblem_solution]

        return solution
    
    def solve_one_trip_sa(self, knapsack, pr=False):

        """
        input:

        knapsack - целочисленное решение задачи о рюкзаке в формате boolean np array

        output: 

        solution - решение задачи развоза подарков для одного трипа в формате boolean np array - 
        в которых содержатся GiftId. GiftId should be ordered by the order of delivery. 

        """

        if pr: print('knapsack', knapsack)
        index=np.where(knapsack==True)[0]
        if pr: print('subproblem index', index)
        if pr: print('length subproblen', len(index))
        subproblem=Problem(self.df_data.iloc[knapsack], None)

        df_solution=subproblem.Greedy_solver()
        
        #вся разница
        #df_solution=self.solve_onetrip_initial(knapsack)
        
        schedule=simulated_annealing(subproblem, df_solution, T=10000, max_iter=200, verbose=False)

        df_solution=schedule.export_to_pandas()
        subproblem_solution=np.array(df_solution['GiftId'])-1
        if pr: print('subproblem_solution', subproblem_solution)

        solution=index[subproblem_solution]

        return solution

    
    def solve_onetrip_initial(self, knapsack, pr=False):  
    
        """
        input:

        knapsack - целочисленное решение задачи о рюкзаке в формате boolean np array

        output: 

        solution - решение задачи развоза подарков для одного трипа в формате boolean np array - 
        в которых содержатся GiftId. GiftId should be ordered by the order of delivery. 

        """
        if pr: print('knapsack', knapsack)
        index=np.where(knapsack==True)[0]
        if pr: print('subproblem index', index)
        if pr: print('length subproblen', len(index))
        subproblem=Problem(self.df_data.iloc[knapsack], None)
        df_solution=subproblem.Greedy_solver_deep()
        
        
        subproblem_solution=np.array(df_solution['GiftId'])-1
        if pr: print('subproblem_solution', subproblem_solution)

        solution=index[subproblem_solution]

        solution=np.arange(0,self.N-1)[knapsack]
        
        d={}
        d['GiftId']=solution
        d['TripId']=[1]*len(solution)
        solution = pd.DataFrame(data=d)      

        return solution


      
    def Randomize_solver(self):
        
        """
        
        output:
        
        rand_sol - решение исходной проблемы в формате dataframe pandas со столбцами GiftId and TripId.
        
        
        """

        gift_weights = self.weights.copy()

        # a list of trip numbers
        rand_trips = []
    
        # a list of gifts ids
        rand_gifts = []

        i = 0
        curr_sleight_weight = 0.
        choices = np.linspace(0,self.N-1,self.N)
        # print(choices)
        # the first point - destination is North poles
        current_gift_destination = self.N-1
        
        while len(choices) != 1:
            # print("len ", len(choices))
            # tmp1 - array of destination of gifts that are the nearest to North pole
            random_gift_destinations = random.choice(choices[:-1])

            current_gift_destination = int(random_gift_destinations)
    
            # add weight to sleight [there no need to calculate cost because of not usage]
            curr_sleight_weight += gift_weights[current_gift_destination]

            if curr_sleight_weight < (self.weight_limit-self.weight_sleighs):

                index = np.where(choices == current_gift_destination)[0]
                choices = np.delete(choices, index)
                rand_gifts.append(current_gift_destination)
                rand_trips.append(i)
                
            else:
                i += 1

                curr_sleight_weight = 0.
                current_gift_destination = self.N-1

            
        # convertation into submision format
        rand_sol = pd.DataFrame({'GiftId':rand_gifts, 'TripId':rand_trips})

        return rand_sol+1
    
    def Greedy_solver(self):
        
        """
        
        output:
        
        greed_solution - решение исходной проблемы в формате dataframe pandas со столбцами GiftId and TripId.
        
        
        """
        # copy the distances in order to change them
        distances = self.dist.copy()
        gift_weights = self.weights.copy()
        
        # make bound for interuptiong the process of finding the trips
        bound = 1e20
        
    
        for i in range(distances.shape[0]):
            distances[i,i] = bound
    
        # a list of trip numbers
        greed_trips = []
    
        # a list of gifts ids
        greed_gifts = []
    
        
        i = 0
        curr_sleight_weight = 0.
        
        # the first point - destination is North poles
        current_gift_destination = self.N-1
        
        while distances[:-1,:-1].min() < bound/2.:
            
            
            # tmp1 - array of destination of gifts that are the nearest to North pole
            nearest_gift_destinations = np.where(distances[:-1, current_gift_destination] == \
                                                 distances[:-1, current_gift_destination].min())[0][0]


            current_gift_destination = nearest_gift_destinations
                 
            # current_gift_destination = giftid
            # add weight to sleight [there no need to calculate cost because of not usage]
            curr_sleight_weight += gift_weights[current_gift_destination]
        
            if curr_sleight_weight < (self.weight_limit-self.weight_sleighs):
                
                # if current weight is small enough we put the bound for current point - gift destination = giftid
                distances[current_gift_destination, :] =  bound 
                # update the set of gifts id f fot current track
                greed_gifts.append(current_gift_destination)
                greed_trips.append(i)
                
            else:
                
                # if we take more gifts then just consider new trip that begins with North pole and 0 weight
                i += 1
                curr_sleight_weight = 0.
                current_gift_destination = self.N-1
                # del greed_trips[-1]

           
                
            
        # convertation into submision format
        greed_solution = pd.DataFrame({'GiftId':greed_gifts, 'TripId':greed_trips})
        # print(greed_gifts)
        
        return greed_solution+1
    
    def one_gift_one_trip_solver(self):
           
        # формируем решение в форме submission
        d = {'GiftId': np.arange(self.N-1)+1, 'TripId': np.arange(self.N-1)}
        solution = pd.DataFrame(data=d)
        
        self.solution=solution
        self.obj=self.objective(self.solution)      
        
        return solution
        

    def Greedy_solver_deep(self):
        
        """
        
        output:
        
        greed_solution - решение исходной проблемы в формате dataframe pandas со столбцами GiftId and TripId.
        
        
        """
        # copy the distances in order to change them
        distances = self.dist.copy()
        gift_weights = self.weights.copy()
        
        # make bound for interuptiong the process of finding the trips
        bound = 1e20
        
    
        for i in range(distances.shape[0]):
            distances[i,i] = bound
    
        # a list of trip numbers
        greed_trips = []
    
        # a list of gifts ids
        greed_gifts = []
    
        
        i = 0
        curr_sleight_weight = 0.
        current_trip_distance = 0.
        #         the first point - destination is North poles
        current_gift_destination = self.N-1
        
        while distances[:-1,:-1].min() < bound/2.:
            
            
            # nearest_gift_destinations - array of destination of gifts that are the nearest to North pole 
            current_destinations_variety = distances[:-1, current_gift_destination] + 0.*current_trip_distance
            loc_obj_func_inonetrip = current_destinations_variety/gift_weights
            nearest_gift_destinations = np.where(loc_obj_func_inonetrip == \
                                                 loc_obj_func_inonetrip.min())[0][0]
            
            # update the trip distance
            current_trip_distance += distances[nearest_gift_destinations, current_gift_destination]
   
            
            current_gift_destination = nearest_gift_destinations
            
       
            # current_gift_destination = giftid
            # add weight to sleight [there no need to calculate cost because of not usage]
            curr_sleight_weight += gift_weights[current_gift_destination]
        
            if curr_sleight_weight < (self.weight_limit-self.weight_sleighs):
                
                # if current weight is small enough we put the bound for current point - gift destination = giftid
                distances[current_gift_destination, :] =  bound 
                # update the set of gifts id f fot current track
                greed_gifts.append(current_gift_destination)
                greed_trips.append(i)
            else:
                
                # if we take more gifts then just consider new trip that begins with North pole and 0 weight
                i += 1
                curr_sleight_weight = 0.
                current_trip_distance = 0.
                current_gift_destination = self.N-1
      
        # convertation into submision format
        greed_solution = pd.DataFrame({'GiftId':greed_gifts, 'TripId':greed_trips})
        
        return greed_solution+1