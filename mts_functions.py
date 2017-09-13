# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:18:36 2017

@author: tiwarir
"""

###############################################################################
# n_city is the number of cities including the central city
###############################################################################

import numpy as np
import math
import pandas as pd


def read_city_lat_long(filename):
    location_data = pd.read_csv(filename, header = None)
    return location_data


def deg2rad(deg):
    return deg * (math.pi/180)  


def get_distance_from_lat_long(lat1,lon1,lat2,lon2):
    R = 6371
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
                math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2))* \
                     math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d
    
def get_distance_matrix(location_data, central_location):
    location_data.loc[-1] = central_location
    location_data.index = location_data.index + 1
    location_data = location_data.sort_index()
    n_city, _ = location_data.shape
    dist_mat = np.zeros([n_city, n_city])
    for i in range(n_city):
        for j in range(i+1, n_city):
            lat1 = location_data.iat[i,0]
            lon1 = location_data.iat[i,1]
            lat2 = location_data.iat[j,0]
            lon2 = location_data.iat[j,1]            
            dist_mat[i,j] = get_distance_from_lat_long(lat1,lon1,lat2,lon2)
            dist_mat[j,i] = dist_mat[i,j]           
    return dist_mat
    

def calculate_total_trip_distance(dist_mat, route, breaks):
    n_city = len(route) 
    n_deliveryboy = len(breaks) + 1
    d = 0
    breaks = np.insert(breaks, 0, 0 )  # add central city at the begining
    breaks = np.append(breaks, n_city)              # add central city at the end
    
    for i in range(n_deliveryboy):
        i_route = route[breaks[i] : breaks[i+1]]
        i_route = np.insert(i_route, 0, 0)   # add central city at the begining
        i_route = np.append(i_route, 0)               # add central city at the end
        i_route_len = len(i_route)
        for j in range(i_route_len - 1):
            d += dist_mat[i_route[j], i_route[j+1]] 
    return d


def calculate_distance_for_whole_population(dist_mat, pop_route, pop_break):
    n, _ = pop_route.shape
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = calculate_total_trip_distance(dist_mat, pop_route[i], pop_break[i])
    return dist
    


def create_temporay_break(n_city, n_deliveryboy):
    temporary_breaks = np.random.permutation(n_city - 1)
    n_breaks = n_deliveryboy - 1
    breaks = np.sort(temporary_breaks[:n_breaks])
    return breaks


def initialize_population(population_size, n_city, n_deliveryboy):
    n_breaks = n_deliveryboy - 1
    population_route = np.zeros((population_size,n_city - 1))
    population_breaks = np.zeros((population_size, n_breaks))
    
    population_route[0,:] = range(1, n_city)
    population_breaks[0,:] = create_temporay_break(n_city, n_deliveryboy)
    
    for i in range(1,population_size):
        population_route[i,:] = np.random.permutation(range(1,n_city))
        population_breaks[i,:] = create_temporay_break(n_city, n_deliveryboy)
        
    population_breaks = population_breaks.astype(int)
    population_route = population_route.astype(int) 
    
    return population_route, population_breaks
    
    
def update_population(population_route, population_breaks, population_distances):
    nrr, ncr = population_route.shape
    nrb, ncb = population_breaks.shape
    new_population_routes = np.zeros((nrr, ncr)) 
    new_population_breaks = np.zeros((nrb, ncb))
    
    random_order = np.random.permutation(nrr)
    
    for p in range(0, nrr, 8):
        ind = random_order[p:p+8]
        routes = population_route[ind,:]
        breaks = population_breaks[ind,:]
        distances = population_distances[ind]
        
        min_ind = np.argmin(distances)
        best_pop_route = routes[min_ind,:]
        best_pop_break = breaks[min_ind, :]
        tmp_pop_routes, tmp_pop_breaks = generate_new_solution_from_the_best_solutions(best_pop_route, best_pop_break)
        new_population_routes[p:p+8,:] = tmp_pop_routes
        new_population_breaks[p:p+8,:] = tmp_pop_breaks
    
    return new_population_routes, new_population_breaks
        
        
    
def generate_new_solution_from_the_best_solutions(best_pop_route, best_pop_break):
    n_city = len(best_pop_route)
    n_deliveryboy = len(best_pop_break)
    insertion_point = np.sort(np.ceil(n_city*np.random.random((2))))
    insertion_point = insertion_point.astype(int)
    
    I = insertion_point[0]
    J = insertion_point[1]
    
    tmp_pop_route = np.zeros((8, n_city))
    tmp_pop_break = np.zeros((8, n_deliveryboy))
    
    # create k solutions
    for k in range(8):
        tmp_pop_route[k,:] = best_pop_route
        tmp_pop_break[k,:] = best_pop_break
        if (k == 1):                # flipping
            ind_1 = range(I,J)
            ind_2 = range(J-1, I-1, -1)
            tmp_pop_route[k, ind_1] = tmp_pop_route[k, ind_2]
        elif ( k == 2):            # swapping 
            tmp_pop_route[k,[I,J]] = tmp_pop_route[k,[J,I]]
        elif (k == 3):            # sliding
            ind_1 = range(I,J)
            ind_2 = range(I+1,J)
            ind_2.append(I)
            tmp_pop_route[k,ind_1] = tmp_pop_route[k,ind_2]
        elif(k == 4 ):         # modify breaks
            tmp_pop_break[k,:] = create_temporay_break(n_city, n_deliveryboy) 
        elif(k == 5): # flip, modify breaks
            ind_1 = range(I,J)
            ind_2 = range(J-1, I-1, -1)
            tmp_pop_route[k, ind_1] = tmp_pop_route[k, ind_2]
            tmp_pop_break[k,:] = create_temporay_break(n_city, n_deliveryboy)            
        elif(k == 6):   # swap, modify breaks
            tmp_pop_route[k,[I,J]] = tmp_pop_route[k, [J,I]]
            tmp_pop_break[k,:] = create_temporay_break(n_city, n_deliveryboy)
        elif (k == 7):
            ind_1 = range(I,J)
            ind_2 = range(I+1,J)
            ind_2.append(I)
            tmp_pop_route[k,ind_1] = tmp_pop_route[k,ind_2]
            tmp_pop_break[k,:] = create_temporay_break(n_city, n_deliveryboy)
    return tmp_pop_route, tmp_pop_break


def run_optimization(population_size, n_deliveryboy, n_iteration, central_location, filename):
    
    location_data = read_city_lat_long(filename)
    
    dist_mat = get_distance_matrix(location_data, central_location)
    
    n_city, _ = location_data.shape
    
    population_route, population_breaks = initialize_population(population_size, n_city, n_deliveryboy)
    
    population_distance = calculate_distance_for_whole_population(dist_mat, pop_route, pop_break)
    
    best_distance = np.zeros(n_iteration)
    best_distance[0] = min(population_distance)
    
    for i in range(1, n_iteration):
        population_route, population_breaks = update_population(population_route, population_breaks, population_distances)
        population_distance = calculate_distance_for_whole_population(dist_mat, pop_route, pop_break)
        best_distance[i] = min(population_distance)
        
    return best_distance, population_route, population_breaks

   
if __name__ == '__main__':
    pop_route, pop_break = initialize_population(10, 8, 3)
    dist_matrix = np.random.randint(100, size=(11, 11)) 
    d = calculate_total_trip_distance(dist_matrix, pop_route[1], pop_break[1])
    population_distances = calculate_distance_for_whole_population(dist_matrix, pop_route, pop_break)
#    central_location = [11.552931,104.933636]
#    location_data = read_city_lat_long('locations.csv')
#    n_salesmen = 25
#    dist_mat = get_distance_matrix(location_data, central_location)
#    n_city, _ = dist_mat.shape
#    p_route = np.random.permutation(range(1,n_city))
#    p_breaks = create_temporay_break(n_city, n_salesmen)

    
