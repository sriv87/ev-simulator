# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:23:26 2024

@author: IRV2KOR
"""
import numpy as np
import pulp as lp
import re
from matplotlib import pyplot as plt
from queue import PriorityQueue
import pandas as pd

def get_arrival_time(num_vehicles):
    if use_case == 1: # Residential use case
        arrival_time = np.random.normal(20,2,num_vehicles)
        return arrival_time
    if use_case == 2: # office use case
        arrival_time = np.random.lognormal(2.32274,0.301833,num_vehicles)
        
        return arrival_time

def get_starting_SoC(is_slow,num_vehicles):
    if is_slow == 1: # Slow charge: one charge per day
        dist_traveled = np.random.lognormal(3.67017251945698, 0.532230403897875, num_vehicles)
        starting_SoC = 1 - dist_traveled/ bat_range
        return starting_SoC
    if is_slow > 1: # Fast charge multiple days per charge
        if use_case == 1 or use_case == 2:
            dist_traveled = np.random.lognormal(3.67017251945698, 0.532230403897875, num_vehicles)
            for i in range(0,num_vehicles):
                dist_traveled[i] = dist_traveled[i]*charg_freq
                if dist_traveled[i] > bat_range*0.8:
                    while True:
                        dist_traveled[i] = np.random.lognormal(3.67017251945698, 0.532230403897875, 1)
                        dist_traveled[i] = dist_traveled[i]*charg_freq
                        if dist_traveled[i] <= bat_range*0.8:
                            break
            starting_SoC = 1 - dist_traveled/bat_range
            return starting_SoC

def time24(time):
    condlist = [time< 2160, time >=2160]
    choicelist= [time, -1]
    time=np.select(condlist, choicelist)
    condlist = [time < 1440, time >= 1440]
    choicelist = [time, time - 1440]
    time = np.select(condlist, choicelist)
   
    return time

def unmanaged_charging(num_vehicles, power, num_chargers):
    iterator = range(nIter)
    load_matrix = np.zeros((nIter, max_time_interval))
    end_SoC = 1
    for it in iterator:
        e_load = np.zeros(max_time_interval)
        start_SoC = get_starting_SoC(is_slow,num_vehicles)
        charg_duration = (end_SoC-start_SoC)*(bat_cap)/(power*chrg_eff)*60
        condlist = [charg_duration < 0, charg_duration >0]
        choicelist = [0,charg_duration]
        charg_duration = np.select(condlist,choicelist)
        
        arrival_time = get_arrival_time(num_vehicles)
        arrival_time.sort()
        arrival_time = np.int_(np.round(arrival_time * 60))
        
        start_time = np.zeros(num_vehicles)
        end_time = np.zeros(num_vehicles)
        start_time[:num_chargers] = arrival_time[:num_chargers]
        end_time[:num_chargers] = np.int_(np.around(start_time[:num_chargers] + charg_duration[:num_chargers]))
        q = PriorityQueue()
        for i in range(0,num_chargers):
            q.put(end_time[i])
        for i in range(num_chargers, len(arrival_time)):
            non_available = [j for j in q.queue if j > arrival_time[i]]
            if len(non_available)==num_chargers:
                start_time[i] = np.int_(min(non_available))+1
                q.get()
            else:
                start_time[i] = np.int_(arrival_time[i])
                q.get()
            end_time[i] = np.int_(np.around(start_time[i] + charg_duration[i]))
            q.put(end_time[i])
        
        wait_time = start_time - arrival_time
        start_time = time24(start_time)
        end_time = time24(end_time)
        start_time = [int(i) for i in start_time]
        end_time = [int(i) for i in end_time]
        for i in range(0,len(wait_time)):
            if end_time[i] == -1:
                wait_time[i] = 0
        avg_wait = np.mean(wait_time)
        unChargedVehicles = end_time.count(-1)
        #print('%4d vehicles uncharged due to lack of chargers' %unChargedVehicles)
        for veh in range(num_vehicles):
            if end_time[veh] == -1 or start_time[veh] == -1:
                break
            if end_time[veh] > start_time[veh]:
                iterator_time = np.arange(start_time[veh]-1, end_time[veh],1)
                for time in iterator_time:
                    e_load[time] += power
            else:
                iterator_time = np.arange(start_time[veh]-1,max_time_interval,1)
                for time in iterator_time:
                    e_load[time] += power
                iterator_time = np.arange(0,end_time[veh],1)
                for time in iterator_time:
                    e_load[time] += power
        load_matrix[it] = e_load
    avg_load = load_matrix.mean(0)
    variance_load = load_matrix.std(0)
    upper_load = avg_load + variance_load**2
    bottom_load = avg_load - variance_load**2
    print('-------------------average wait-----------------',avg_wait)
    total_soc = sum(start_SoC)
    print('total unmanaged SOC:', total_soc)
    
    return avg_load, bottom_load, upper_load, avg_wait
                    
def unmanaged_usecase_setup(useCase):
    avg_wait_slow = -1
    avg_wait_fast = -1 
    slow_load_1h_avg = np.zeros(24)
    fast_load_1h_avg = np.zeros(24)
    timeRange = range(24)
    
    if use_case == 2:
        avg_load, bottom_load, upper_load, avg_wait = unmanaged_charging(num_fast_chrg_vehicles, power_v0g_fast, num_fast_chargers)
        avg_wait_fast = avg_wait
        for time in timeRange:
            fast_load_1h_avg[time] = np.average(avg_load[time*60:(time+1)*60]) # Converting EV loads into one hour loads to superimpose with building loads
        avg_load, bottom_load, upper_load, avg_wait = unmanaged_charging(num_slow_chrg_vehicles, power_v0g_slow, num_slow_chargers)
        avg_wait_slow = avg_wait
        for time in timeRange:
            slow_load_1h_avg[time] = np.average(avg_load[time*60:(time+1)*60]) # Converting EV loads into one hour loads to superimpose with building loads
    if use_case == 1:
        avg_load, bottom_load, upper_load, avg_wait = unmanaged_charging(num_slow_chrg_vehicles, power_v0g_slow, num_slow_chargers)
        avg_wait_slow = avg_wait
        for time in timeRange:
            slow_load_1h_avg[time] = np.average(avg_load[time*60:(time+1)*60]) # Converting EV loads into one hour loads to superimpose with building loads
        
        avg_load, bottom_load, upper_load, avg_wait = unmanaged_charging(num_fast_chrg_vehicles, power_v0g_fast, num_fast_chargers)
        avg_wait_fast = avg_wait
        for time in timeRange:
            fast_load_1h_avg[time] = np.average(avg_load[time*60:(time+1)*60]) # Converting EV loads into one hour loads to superimpose with building loads
    print('----------------------',avg_wait_slow,avg_wait_fast)
    return avg_wait_slow, avg_wait_fast, slow_load_1h_avg, fast_load_1h_avg

def initiate_v1g(num_v1g_vehicles):
    v1g_start_time = get_arrival_time(num_v1g_vehicles)
    condlist = [v1g_start_time < 24, v1g_start_time >= 24]
    choicelist = [v1g_start_time, v1g_start_time-24]
    start_time = np.select(condlist,choicelist)
    v1g_start_time = np.int_(np.round(start_time))
    print('----starting time----', v1g_start_time)
    temp_start = v1g_start_time
    temp_park_random = np.zeros(num_v1g_vehicles)
    for vehicle in range(num_v1g_vehicles):
        temp_park = parking_duration[temp_start[vehicle]-1]
        temp_park_random[vehicle] = np.random.normal(temp_park,0.5,1)
    v1g_end_time = temp_start+temp_park_random
    condlist = [v1g_end_time < 24, v1g_end_time >= 24]
    choicelist = [v1g_end_time,v1g_end_time-24]
    end_time = np.select(condlist,choicelist)
    v1g_end_time = np.int_(np.round(end_time))
    
    # starting Soc and travel distance fr each vehciele
    v1g_start_SoC = get_starting_SoC(is_slow,num_v1g_vehicles)
    totalsoc=sum(v1g_start_SoC)
    print('total v1g soc:', totalsoc)
    return v1g_start_time, v1g_end_time,v1g_start_SoC


    
def v1g_optimize(num_v1g_vehicles):
    # initiate the problem statement
    model=lp.LpProblem('Minimize_Distribution_level_Peak_Valley_Difference', lp.LpMinimize)

    #define optimization variables
    veh_V1G=range(num_v1g_vehicles)
    time_Interval=range(24)

    charge_profiles=lp.LpVariable.dicts('charging_profiles', ((i,j) for i in veh_V1G for j in time_Interval))
    charge_states=lp.LpVariable.dicts('charging_states', ((i,j) for i in veh_V1G for j in time_Interval), cat='Binary')
    total_load=lp.LpVariable.dicts('total_load', time_Interval,lowBound=0.1)
    max_load=lp.LpVariable('max_load', lowBound=0)
    min_load=lp.LpVariable('min_load', lowBound=0)

    # define objective function
    #model += max_load - min_load
    model += max_load

    # define constraints
    for t in time_Interval: # constraint 1 & 2: to identify the max and min loads
        model += total_load[t] <= max_load
        model += total_load[t] >= min_load

    for t in time_Interval: # constraint 3: calculate the total load at each time interval t
        model += lp.lpSum( [charge_profiles[i,t]] for i in veh_V1G) + base_load[t] + unmanaged_load[t] == total_load[t]

    for i in veh_V1G: # constraint 4: constraint on charging powers for each EV: only optimize the charge profile between start and end charging time
        temp_start=v1g_start_time[i]
        temp_end=v1g_end_time[i]
        if temp_start >= temp_end:
            for t in range (temp_end):
                model += charge_profiles[i,t] <= charge_states[i,t] * power_v1g_upper_bound
                model += charge_profiles[i,t] >= charge_states[i,t] * power_v1g_lower_bound
            for t in range(temp_end, temp_start, 1):
                model += charge_profiles[i,t] == 0
                model += charge_states[i,t] == 0
            for t in range(temp_start, 24, 1):
                model += charge_profiles[i,t] <= charge_states[i,t] * power_v1g_upper_bound
                model += charge_profiles[i,t] >= charge_states[i,t] * power_v1g_lower_bound
        if temp_start < temp_end:
            for t in range(temp_start):
                model += charge_profiles[i,t] == 0
                model += charge_states[i,t] ==0
            for t in range(temp_start, temp_end, 1):
                model += charge_profiles[i,t] <= charge_states[i,t] * power_v1g_upper_bound
                model += charge_profiles[i,t] >= charge_states[i,t] * power_v1g_lower_bound
            for t in range(temp_end, 24, 1):
                model += charge_profiles[i,t] == 0
                model += charge_states[i,t]==0

    for i in veh_V1G: # constraint 5: SOC constraint, cannot be greater than 1, end_SOC must be above certain levels
        temp_start=v1g_start_time[i]
        temp_end=v1g_end_time[i]
        temp_startSOC=v1g_start_SoC[i]
        if temp_start >= temp_end:
            for t in range(temp_start+1, 24, 1):
                temp_timer = range (temp_start, t, 1)
                model += temp_startSOC + lp.lpSum( [charge_profiles[i,tn] *chrg_eff/bat_cap] for tn in temp_timer) <=1
            for t in range (0, temp_end+1, 1):
                temp_timer = range (0, t, 1)
                model += temp_startSOC + lp.lpSum( [charge_profiles[i,tn] *chrg_eff/bat_cap] for tn in range(temp_start, 24,1)) \
                         + lp.lpSum( [charge_profiles[i,tn] *chrg_eff/bat_cap] for tn in temp_timer) <=1
                
            #if end_SOC == 1:
            #    incrementSOC=v1g_distance[i]/batteryRange
            #    model += lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in  range(temp_start, 24, 1)) \
            #             + lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in temp_timer) >= incrementSOC  # need to divide 4
            if end_SoC ==2:
                model += temp_startSOC + lp.lpSum([charge_profiles[i, tn] *chrg_eff/bat_cap] for tn in range(temp_start, 24, 1)) + lp.lpSum([charge_profiles[i, tn] *chrg_eff/bat_cap] for tn in range(0, temp_end)) ==1
        if temp_start < temp_end:
             for t in range (temp_start+1, temp_end+1, 1):
                temp_timer = range (temp_start, t, 1)
                model += temp_startSOC + lp.lpSum( [charge_profiles[i,tn] *chrg_eff/bat_cap] for tn in temp_timer) <=1
             #if end_SOC == 1:
             #   incrementSOC=v1g_distance[i]/batteryRange
             #   model += lp.lpSum( [chargeprofiles[i,tn] * charge_efficiency/batteryCapacity] for tn in temp_timer) >= incrementSOC
             if end_SoC ==2:
                #model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] *(1/charge_efficiency / batteryCapacity)] for tn in range(temp_start, 24, 1)) + lp.lpSum([chargeprofiles[i, tn] *(1/charge_efficiency / batteryCapacity)] for tn in temp_timer) ==1
                model += temp_startSOC + lp.lpSum([charge_profiles[i, tn] * chrg_eff/bat_cap] for tn in range(temp_start, temp_end, 1)) ==1

                #for t in time_Interval:  # constraint 6: number of chargers available at time interval t
        #model += lp.lpSum([chargestates[i, t]] for i in veh_V1G) <= 400

   
    status=model.solve()
    model.writeLP("v1g.lp")
    
    print(lp.LpStatus[status])
    print(lp.value(max_load), lp.value(min_load))
    return charge_profiles, total_load

def initiate_v2g(num_v2g_vehicles):
    v2g_start_time = get_arrival_time(num_v2g_vehicles)
    condlist = [v2g_start_time < 24, v2g_start_time >= 24]
    choicelist = [v2g_start_time, v2g_start_time-24]
    start_time = np.select(condlist,choicelist)
    v2g_start_time = np.int_(np.round(start_time))
    temp_start = v2g_start_time
    temp_park_random = np.zeros(num_v2g_vehicles)
    for vehicle in range(num_v2g_vehicles):
        temp_park = parking_duration[temp_start[vehicle]-1]
        temp_park_random[vehicle] = np.random.normal(temp_park,0.5,1)
    v2g_end_time = temp_start+temp_park_random
    condlist = [v2g_end_time < 24, v2g_end_time >= 24]
    choicelist = [v2g_end_time,v2g_end_time-24]
    end_time = np.select(condlist,choicelist)
    v2g_end_time = np.int_(np.round(end_time))
    v2g_start_SoC = get_starting_SoC(2,num_v2g_vehicles)
    return v2g_start_time, v2g_end_time,v2g_start_SoC

def v2g_optimize(num_v2g_vehicles):
    # initiate the problem statement
    model=lp.LpProblem('Minimize_Distribution_level_Peak_Valley_Difference', lp.LpMinimize)

    #define optimization variables
    veh_V2G=range(num_v2g_vehicles)
    time_Interval=range(24)
    chargeprofiles=lp.LpVariable.dicts('charging_profiles', ((i,j) for i in veh_V2G for j in time_Interval))
    chargestates=lp.LpVariable.dicts('charging_states', ((i,j) for i in veh_V2G for j in time_Interval), cat='Binary')
    dischargeprofiles=lp.LpVariable.dicts('discharging_profiles', ((i,j) for i in veh_V2G for j in time_Interval))
    dischargestates=lp.LpVariable.dicts('discharging_states', ((i,j) for i in veh_V2G for j in time_Interval), cat='Binary')
    total_load=lp.LpVariable.dicts('total_load', time_Interval,lowBound=0)
    max_load=lp.LpVariable('max_load', lowBound=0)
    min_load=lp.LpVariable('min_load', lowBound=0)
    
    
    if optimization_obj == 1:
        # define objective function
        #model += max_load 
        model += max_load - min_load
    elif optimization_obj == 2:
        model += lp.lpSum((total_load[t]*utility_cost[t] for t in time_Interval))
        
# =============================================================================
#         for t in time_Interval:
#             utility_cost_optimization = lp.lpSum(total_load[t]*utility_cost[t])
#         model += utility_cost_optimization
# =============================================================================
    # define constraints
    for t in time_Interval: # constraint 1 & 2: to identify the max and min loads
        
        model += total_load[t] <= max_load
        model += total_load[t] >= min_load

    for t in time_Interval: # constraint 3: calculate the total load at each time interval t
        model += lp.lpSum([chargeprofiles[i,t]] for i in veh_V2G) + lp.lpSum([dischargeprofiles[i,t]*dischrg_eff for i in veh_V2G]) + base_load[t] + unmanaged_load[t] == total_load[t] #need to plus base loads

    for i in veh_V2G: # constraint 4: constraint on charging powers for each EV: only optimize the charge profile between start and end charging time
        temp_start=v2g_start_time[i]
        temp_end=v2g_end_time[i]
        if temp_start >= temp_end:
            for t in range (temp_end):
                model += chargestates[i,t] + dischargestates[i,t] <=1
                model += chargeprofiles[i,t] <= chargestates[i,t] * power_v1g_upper_bound
                model += chargeprofiles[i,t] >= chargestates[i,t] * power_v1g_lower_bound
                model += dischargeprofiles[i,t] <= dischargestates[i,t] * power_v2g_upper_bound
                model += dischargeprofiles[i,t] >= dischargestates[i,t] * power_v2g_lower_bound
            for t in range(temp_end, temp_start, 1):
                model += chargeprofiles[i,t] == 0
                model += chargestates[i,t] == 0
                model += dischargeprofiles[i,t]==0
                model += dischargestates[i,t]==0
            for t in range(temp_start, 24, 1):
                model += chargestates[i, t] + dischargestates[i, t] <= 1
                model += chargeprofiles[i,t] <= chargestates[i,t] * power_v1g_upper_bound
                model += chargeprofiles[i,t] >= chargestates[i,t] * power_v1g_lower_bound
                model += dischargeprofiles[i, t] <= dischargestates[i, t] * power_v2g_upper_bound
                model += dischargeprofiles[i, t] >= dischargestates[i, t] * power_v2g_lower_bound

        if temp_start < temp_end:
            
            for t in range(temp_start):
                model += chargeprofiles[i,t] == 0
                model += chargestates[i,t] ==0
                model += dischargeprofiles[i,t]==0
                model += dischargestates[i,t]==0
            for t in range(temp_start, temp_end, 1):
                model += chargestates[i, t] + dischargestates[i, t] <= 1
                model += chargeprofiles[i,t] <= chargestates[i,t] * power_v1g_upper_bound
                model += chargeprofiles[i,t] >= chargestates[i,t] * power_v1g_lower_bound
                model += dischargeprofiles[i, t] <= dischargestates[i, t] * power_v2g_upper_bound
                model += dischargeprofiles[i, t] >= dischargestates[i, t] * power_v2g_lower_bound
            for t in range(temp_end, 24, 1):
                model += chargeprofiles[i,t] == 0
                model += chargestates[i,t]==0
                model += dischargeprofiles[i,t]==0
                model += dischargestates[i,t]==0

    for i in veh_V2G: # constraint 5: SOC constraint, cannot be greater than 1, end_SOC must be above certain levels
        temp_start=v2g_start_time[i]
        temp_end=v2g_end_time[i]
        temp_startSOC=v2g_start_SoC[i]
        if temp_start >= temp_end:
            for t in range(temp_start+1, 24, 1):
                temp_timer = range (temp_start, t, 1)
                model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] *chrg_eff/bat_cap] for tn in temp_timer) \
                         + lp.lpSum( [dischargeprofiles[i,tn] *(1/bat_cap)] for tn in temp_timer) <=1 #need to divide 4!
            for t in range (0, temp_end+1, 1):
                temp_timer = range (0, t, 1)
                model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] * chrg_eff/bat_cap] for tn in range(temp_start, 24,1)) + lp.lpSum( [chargeprofiles[i,tn] * chrg_eff/bat_cap] for tn in temp_timer) \
                         + lp.lpSum( [dischargeprofiles[i,tn] *(1/bat_cap)] for tn in range(temp_start, 24,1)) + lp.lpSum( [dischargeprofiles[i,tn] *(1/bat_cap)] for tn in temp_timer) <=1 #need to divide 4
            #if end_SOC == 1:
            #    incrementSOC=v2g_distance[i]/batteryRange
            #    model += lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in  range(temp_start, 24, 1)) + lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in temp_timer) \
            #             + lp.lpSum([dischargeprofiles[i, tn] *(1/ batteryCapacity)] for tn in  range(temp_start, 24, 1)) + lp.lpSum([dischargeprofiles[i, tn] *(1/ batteryCapacity)] for tn in temp_timer) >= incrementSOC  # need to divide 4
            if end_SoC ==2:
                model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] * chrg_eff / bat_cap] for tn in range(temp_start, 24, 1)) + lp.lpSum([chargeprofiles[i, tn] * chrg_eff / bat_cap] for tn in temp_timer) \
                         + lp.lpSum([dischargeprofiles[i, tn] *(1/ bat_cap)] for tn in range(temp_start, 24, 1)) + lp.lpSum([dischargeprofiles[i, tn] *(1/ bat_cap)] for tn in temp_timer) ==1

        if temp_start < temp_end:
             for t in range (temp_start+1, temp_end+1, 1):
                temp_timer = range (temp_start, t, 1)
                model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] * chrg_eff /bat_cap] for tn in temp_timer) \
                         + lp.lpSum( [dischargeprofiles[i,tn] *(1/bat_cap)] for tn in temp_timer) <=1 #need to divide by 4
             #if end_SOC == 1:
             #   incrementSOC=v2g_distance[i]/batteryRange
             #   model += lp.lpSum( [chargeprofiles[i,tn] * charge_efficiency/batteryCapacity] for tn in temp_timer)\
             #            +lp.lpSum( [dischargeprofiles[i,tn] *(1/batteryCapacity)] for tn in temp_timer) >= incrementSOC  # need to divide 4
             if end_SoC ==2:
                 
# =============================================================================
#                 model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in range(temp_start, 24, 1)) + lp.lpSum([chargeprofiles[i, tn] * charge_efficiency / batteryCapacity] for tn in temp_timer)\
#                          + lp.lpSum([dischargeprofiles[i, tn] *(1/batteryCapacity)] for tn in range(temp_start, 24, 1)) + lp.lpSum([dischargeprofiles[i, tn] *(1/batteryCapacity)] for tn in temp_timer)==1

# =============================================================================
                 model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] * chrg_eff / bat_cap] for tn in range(temp_start, temp_end, 1))\
                          + lp.lpSum([dischargeprofiles[i, tn] *(1/bat_cap)] for tn in range(temp_start, temp_end, 1)) ==1
    #print(model)

    #status=model.solve(lp.COIN(maxSeconds=2500))
    status=model.solve()
    print(lp.LpStatus[status])
    print(lp.value(max_load))

    return chargeprofiles, dischargeprofiles, total_load
       
def loadAnalysis():
    if charging_strategy == 1:
        optimized_ev_profiles = np.zeros((num_v1g_vehicles,24))
    elif charging_strategy ==2:
        optimized_ev_profiles = np.zeros((num_v2g_vehicles,24))
    for item in charge_profiles.items():
        name = item[1].name
        index = re.findall(r'\d+', name)
        index = [int(i) for i in index]
        veh_index = index[0]
        time_index = index[1]
        optimized_ev_profiles[veh_index][time_index] = item[1].varValue
    if charging_strategy == 2:
        for item in discharge_profiles.items():
            name = item[1].name
            index = re.findall(r'\d+', name)
            index = [int(i) for i in index]
            veh_index = index[0]
            time_index = index[1]
            optimized_ev_profiles[veh_index][time_index] = item[1].varValue + optimized_ev_profiles[veh_index][time_index]
    
    optimized_ev_load = np.zeros(24)
    for i in range(24):
        optimized_ev_load[i] = sum(row[i] for row in optimized_ev_profiles)
    optimized_total_load = np.zeros(24)
    for item in total_load.items():
        name = item[1].name
        index = re.findall(r'\d+', name)
        index = [int(i) for i in index]
        time_index = index[0]
        optimized_total_load[time_index] = item[1].varValue
        
    if charging_strategy == 0: #unmanaged charging
        if max(slow_load_1h_avg) > 0:
            slow_coincidence = (max(slow_load_1h_avg)/power_v0g_slow)/num_slow_chrg_vehicles
            print('Coincidence factor for slow charging(unmanaged) is', "{:.0%}".format(slow_coincidence))
        if max(fast_load_1h_avg) > 0:
            fast_coincidence = (max(fast_load_1h_avg)/power_v0g_fast)/num_fast_chrg_vehicles
            print('Coincidence factor for fast charging(unmanaged) is', "{:.0%}".format(fast_coincidence))
        if avg_wait_slow >= 0:
            print('Average wait time for slow chargers(unmanaged) is %5.2f hours' % avg_wait_slow)
        if avg_wait_fast >= 0:
            print('Average wait time for slow chargers(unmanaged) is %5.2f hours' % avg_wait_fast)
            
    capacity_limit_line = np.full(24,transformer_cap)
    efficiency_line = np.full(24,transformer_eff*transformer_cap)
        
    x = np.arange(0, 24, 1)
# =============================================================================
#     plt.rc('font', family='serif')
#     fig = plt.figure(figsize=(4, 3))
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True,figsize=(15, 6))
#     ax1.set_xlabel('Time of a day: hour')
#     ax1.set_ylabel('Load: kW')
#     ax1.fill_between(x, unmanaged_load, color='whitesmoke', label='EV Load: Unmanaged Charging')
#     ax1.fill_between(x, optimized_ev_load, color='lavender', label='EV Load: V2G')
#     EV_load = optimized_ev_load + unmanaged_load
#     ax1.plot(x, EV_load, color='orange', lw=3, label='EV Load: Total')
#     ax1.legend(fontsize=12, loc='upper left')
#     ax1.set_ylim([-1000, 4500])
# 
#     ax2.set_xlabel('Time of a day: hour')
#     ax2.set_ylabel('Load: kW')
#     ax2.set_ylim([-1000, 4500])
#     ax2.plot(x, optimized_total_load, color='gray', lw=3, label='Total load')
#     ax2.fill_between(x, base_load, optimized_total_load, color='orange', label='EV Load')
#     ax2.fill_between(x, base_load, color='yellow', label='Base Load')
#   
# 
#     ax2.plot(x, capacity_limit_line, 'r:', label='Rated capacity')
#     ax2.plot(x, efficiency_line, 'y:', label='80% of Rated capacity')
#     ax2.legend(fontsize=12, loc='upper left')
# =============================================================================
    plt.rc('font', family='serif')
    #fig = plt.figure(figsize=(4, 3))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.set_xlabel('Time of a day: hour')
    ax1.set_ylabel('Load: kW')
    ax1.fill_between(x, unmanaged_load, color='red', label='EV Load: Unmanaged Charging')
    ax1.fill_between(x, optimized_ev_load, color='lightcyan', label='EV Load: V2G')
    EV_load = optimized_ev_load + unmanaged_load
    ax1.plot(x, EV_load, color='orange', lw=3, label='EV Load: Total')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_ylim([-1000, 4500])

    ax2.set_xlabel('Time of a day: hour')
    ax2.set_ylabel('Load: kW')
    ax2.set_ylim([-1000, 4500])
    ax2.plot(x, optimized_total_load, color='gray', lw=3, label='Total load')
    ax2.fill_between(x, base_load, optimized_total_load, color='orange', label='EV Load')
    ax2.fill_between(x, base_load, color='lemonchiffon', label='Base Load')
    ax2.plot(x, capacity_limit_line, 'r:', label='Rated capacity')
    ax2.plot(x, efficiency_line, 'y:', label='80% of Rated capacity')
    ax2.legend(fontsize=8, loc='upper left')
    
    max_load = max(optimized_total_load)
    max_base_load = max(base_load)
    delta_max_load = max_load - max_base_load
    max_index = np.argmax(optimized_total_load)
    max_time = max_index+1 
    ev_to_peak_load = (optimized_ev_load[max_index]+unmanaged_load[max_index])/max_load
    max_load_factor = max_load/transformer_cap
    increased_cap = max_load - transformer_eff*transformer_cap
    min_load = min(optimized_total_load)
    peak_valley_diff = max_load-min_load
    max_ev_load = max(EV_load)
    
    print('Maximum of EV load: {} kW'.format(np.round(max_ev_load,1)))
    print('Peak load: {} kW'.format(np.round(max_load, 1)))
    print('Increases in peak load: {} kW'.format(np.round(delta_max_load, 1)))
    print('Peak valley difference: {} kW'.format(np.round(peak_valley_diff, 1)))
    if charging_strategy==0:
        print('EV ratio in peak load: {:.0%}'.format(np.round(ev_to_peak_load, 2)))
        print('Time of the peak load', max_time)
    print('Utilization factor: {:.0%}'.format(np.round(max_load_factor, 2)))
    print('Increased capacity required: {} kW'.format(np.round(increased_cap, 0)))

    return EV_load,optimized_total_load
if __name__ == '__main__':
    
    """ 
    Defining inputs to the charging calculator
    
    """
    ## Flags for different settings
    use_case =2 # Flag for different usecases: 1. Residential, 2. Office. 3. Public (not-implemented)
    charging_strategy = 2 # 0 - unmanaged only, 1 - managed v1g (includes unmanaged), 2 - managed v2g(includes unmanaged)
    is_slow = 1# slow charging
    optimization_obj =2 # 1 - peak shaving, 2 - utility cost minimization
     
    
    # Initiate the number and enrollment of vehicles
    num_vehicles = np.int_(800) # Total number of EVs to be simulated
    if charging_strategy  == 0:
        percent_v1g = 0  # percentage of vehicles enrolled in V1G
        percent_v0g = 1
        percent_v2g = 0
        if use_case == 1:
            percent_fast_chrg_vehicles = 0 # Number of vehicles using fast chargers in unmanaged charging
            num_slow_chargers = num_vehicles # by default equal to the number of vehicles
            num_fast_chargers = 0 # fixed. Assumed no fast charges in homes
            power_v0g_slow = 7
            power_v0g_fast = 0 
            power_v1g_upper_bound = 0  
            power_v1g_lower_bound = 0   
        if use_case == 2:
            percent_fast_chrg_vehicles = 1
            num_slow_chargers = 0 #fixed. Assumed no fast charges in homes
            num_fast_chargers = num_vehicles # by default equal to the number of vehicles
            power_v0g_fast = 20 
            power_v0g_slow = 0
            power_v1g_upper_bound = 0  
            power_v1g_lower_bound = 0  
    elif charging_strategy == 1:
        percent_v1g = 0.5  # percentage of vehicles enrolled in V1G. Can take a value between 0 and 1
        percent_v0g = 1-percent_v1g
        percent_v2g = 0
        if use_case == 1:
            percent_fast_chrg_vehicles = 0 # Number of vehicles using fast chargers in unmanaged charging
            num_slow_chargers = 100 # by default equal to the number of vehicles 
            num_fast_chargers = 0 # fixed. Assumed no fast charges in homes
            power_v0g_slow = 3
            power_v0g_fast = 0 
            power_v1g_upper_bound =20
            power_v1g_lower_bound = 3  
        if use_case == 2:
            percent_fast_chrg_vehicles = 1
            num_slow_chargers = 0 # 
            num_fast_chargers = 100 # 
            power_v0g_fast = 20 
            power_v0g_slow = 0
            power_v1g_upper_bound = 20    
            power_v1g_lower_bound = 3  
    elif charging_strategy == 2:
        percent_v2g = 0.2
        percent_v1g = 0
        percent_v0g = 1-percent_v2g
        if use_case == 1:
            percent_fast_chrg_vehicles = 0 # Number of vehicles using fast chargers in unmanaged charging
            num_slow_chargers = 50 # 
            num_fast_chargers = 0 # fixed. Assumed no fast charges in homes
            power_v0g_slow = 7
            power_v0g_fast = 0 
            power_v1g_upper_bound = 7  
            power_v1g_lower_bound = 3  
            power_v2g_upper_bound = -3 
            power_v2g_lower_bound= -7 
        if use_case == 2:
            percent_fast_chrg_vehicles = 0
            num_slow_chargers = 512# 
            num_fast_chargers = 512 # 
            power_v0g_slow = 7
            power_v0g_fast = 10 
            power_v2g_upper_bound = -3 
            power_v2g_lower_bound= -7 
            power_v1g_upper_bound = 7
            power_v1g_lower_bound = 3  
    
    
    num_v2g_vehicles = np.int_(np.round(num_vehicles*percent_v2g))
    num_v1g_vehicles = np.int_(np.round(num_vehicles*percent_v1g))
    num_v0g_vehicles = np.int_(np.round(num_vehicles*percent_v0g))
    num_fast_chrg_vehicles = np.int_(np.round(num_v0g_vehicles*percent_fast_chrg_vehicles))
    print('-----------------------',num_fast_chrg_vehicles)
    num_slow_chrg_vehicles = np.int_(np.round(num_v0g_vehicles*(1-percent_fast_chrg_vehicles)))
    if (use_case == 1 or use_case == 2) and (charging_strategy==2):
        if num_fast_chrg_vehicles > 0:
            num_fast_chrg_vehicles = np.int_(np.round(num_fast_chrg_vehicles*0.8))
            
    
    # EVs' tech specifications
    bat_cap = 45 # Battery capacity of the EVs in [kWh]    
    veh_eff = 15 # Venhicles' energy efficiency in kWh/100 km
    bat_range= bat_cap/veh_eff * 100
    end_SoC = 2 #Flag for end SoC: 1 - Charrging to cover next travel, 2: Charge fully 
    
    
    
    if num_fast_chargers > num_fast_chrg_vehicles:
        print('does it come here---------------')
        num_fast_chargers = num_fast_chrg_vehicles
        print(num_fast_chrg_vehicles)
    if num_slow_chargers > num_slow_chrg_vehicles:
        num_slow_chargers = num_slow_chrg_vehicles
    
    #Charging specs
    chrg_eff = 0.9 # efficiency of charging
    dischrg_eff = 0.9 # efficiency of discharging
    if charging_strategy == 2:
        charg_freq= 1 # Charge frequency
    else:
        charg_freq=3
    
      
    # Transformer capacity and base load input
    transformer_cap = 4000 # Rated capacity of the transformer in KVa
    if use_case == 1:
        base_load = [1578.7, 1414.7, 1290.1, 1258.6, 1199.2, 1279.8, 1327.9, 1378, 1492.4, 1666.8, 1738.2, 1497.9, 1433.9,
                     1446.8, 1463.8, 1434.5, 1523.9, 1651.1, 1727.2, 1922, 2162.6, 2192.6, 1944.4, 1762.9]# base load in kW
    if use_case == 2:
        base_load =np.array([1043.071385, 1009.268146, 980.293941, 985.1229752, 973.0503897, 1011.682663, 1294.181163, 1675.674865, 2087.350029,2290.169466, 2391.579184, 2125.982303,2237.050089, 1812.095081, 1746.903119,1733.623275, 1717.928914, 1800.022495,1558.570785, 1302.631973, 1154.139172,981.5011995, 874.0551888,	802.8269344])
    transformer_eff = 0.8
    
    # Utility pricing specs
    utility_cost = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05]
    # Sim characteristics
    nIter = 25 # number of monte carlo iterations
    max_time_interval =1440 #time interval for unmanaged model
    if (charging_strategy == 1 or charging_strategy == 0) and (use_case == 1 or use_case==2):
        parking_duration=np.zeros(24) + 8 #parking duration for managed charging
    elif (charging_strategy ==2) and (use_case==1 or use_case==2):
        print('----------------------------------------------------------#####################')
        parking_duration = [9.8,9,9.3,8.8,8.5,7.3,7.4,7.7,6.8,5.4, 5, 5.2, 5.1, 5.2, 5, 6.1, 7.2, 9.1, 9.8, 9.8, 10.7, 9.4, 10.1, 8.8]
    
    unmanaged_load=np.zeros(24)
    
    #Unmanaged charging results
    avg_wait_slow, avg_wait_fast, slow_load_1h_avg, fast_load_1h_avg = unmanaged_usecase_setup(use_case)
    unmanaged_load = slow_load_1h_avg+fast_load_1h_avg
    print(unmanaged_load)
    if charging_strategy == 1:
        
        v1g_start_time, v1g_end_time, v1g_start_SoC = initiate_v1g(num_v1g_vehicles)
        charge_profiles,total_load = v1g_optimize(num_v1g_vehicles)   
    elif charging_strategy==2:
        v2g_start_time, v2g_end_time, v2g_start_SoC = initiate_v2g(num_v2g_vehicles)
        charge_profiles,discharge_profiles,total_load = v2g_optimize(num_v2g_vehicles)
     
    EV_load,optimized_total_load = loadAnalysis()
    
    #avgwait, oneHourAvgLoad = unManaged(useCase)
    
   