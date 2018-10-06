'''
 * Author:    Sokratis Kouvaras
 * Created:   19.09.2018
 * Dependencies: numpy -> 1.14.3
 *               matplotlib -> 2.2.2
 *
 * Description: This source file contains a class that reads files in the TSP
 * format and uses the meta-heuristic appoach(Variable Neighbourhood Descend)
 * to solve the Travelling Salesman Problem.
 * 
 * Files : Files on TSP Format can be found at:
 * http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
 *
 * (c) Copyright 2018 , Sokratis Kouvaras, All rights reserved.
 **/
'''
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import urllib
from bs4 import BeautifulSoup


def distance(item1,item2):
    k=math.sqrt(((item1[0]-item2[0])**2)+(item1[1]-item2[1])**2)    
    return k   




class VND :
    def __init__(self,filepath):
        if filepath[:3]=='htt':
            self.stops = self.parse_http(filepath)
        elif filepath[-3:]=="tsp":
            self.stops = self.parse_tsp(filepath)
        self.N = len(self.stops)
        self.distance_matrix = np.zeros(shape=(self.N,self.N))
        for i in range(0,len(self.stops)):
            for j in range(0,len(self.stops)):
                s=distance(self.stops[i],self.stops[j])
                self.distance_matrix[i][j]=s
        # A simple menu to initialize the route
        print("Choose a construction method to initialize a route:\n(1):Random route\n(2):G.R.A.S.P method\n(3):Nearest Neighbour method\n(4):Double-ended Nearest Neighbour method")
        choice=int(input("Your choice : "))
        if choice==1:
            self.route = self.construction_using_shuffle()
            print(self.total_route_distance())
            self.image(self.route,self.stops)
            self.global_best = self.total_route_distance()
        elif choice==2:
            self.rcl_length=input("Size of restricted candidate list:")
            self.route=self.construction_using_GRASP_method(int(self.rcl_length))
            print(self.total_route_distance())
            self.global_best = self.total_route_distance()
            self.image(self.route,self.stops)

        elif choice==3:
            self.route = self.construction_using_nearest_neighbour()
            print(self.total_route_distance())
            self.image(self.route,self.stops)
            self.global_best = self.total_route_distance()
        elif choice==4:
            self.route = self.construction_using_double_nearest_neighbour()
            print(self.total_route_distance())
            self.image(self.route,self.stops)
            self.global_best = self.total_route_distance()
        else:
            raise ValueError("Invalid Input")    
        self.best_route = self.route

######################## Reading the TSP Format             ########################################
    #This method parses the TSPLib format
    def parse_http(self,filepath):
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html,'lxml')
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text().split()
        for i in range(len(text)):
            if text[0] == 'DIMENSION:':
                N = text [i+1]
                del(text[0])
                del(text[0])
                break
            del(text[0])
        while text[0] != 'NODE_COORD_SECTION':
            del(text[0])
        del(text[0])
        stops = []
        while text[0] != 'EOF':
            del(text[0])
            stops.append( (float(text[0]),float(text[1]) ) )
            del(text[0])
            del(text[0])
        return stops
                
    def parse_tsp(self,filepath):    
        cities_set=[]
        cities_tups=[]
        stops=[]
        n=[]
        m=[]
        Q=0
        f= open(str(filepath), 'r')
        content = f.read().splitlines()
        cleaned = [x.lstrip() for x in content if x != ""] 
        for item in cleaned:
            if item.startswith("DIMENSION"):
                N=int(re.compile(r'[^\d]+').sub("",item))                                                                                                                         
        for item in cleaned:
            for i in range(1, N+1):                
                if item.startswith(str(i)):        
                    rest=item.partition(' ')        
                    if rest not in cities_set:     
                        cities_set.append(rest) 
        for i in range(0,len(cities_set)):             
            cities_tups.append(cities_set[i][2])       
        for j in range(0,len(cities_tups)):            
            for i in range(0,len(cities_tups[j])):     
                if cities_tups[j][i]!=" ":            
                    Q=Q+1                          
                    m.append(cities_tups[j][i])
                else:
                    break
            for i in range(Q+1,len(cities_tups[j])):     
                n.append(cities_tups[j][i])   
            b = ''.join(m) 
            z=''.join(n)
            v=float(b)     
            W=float(z)
            stops.append((v,W))     
            Q=0
            m=[]
            n=[]
        return stops
######################## Initial Route Construction Methods ########################################    
    # This method finds the closest N neighbours of a given stop
    # This method is required to initialize a route with G.R.A.S.P and N.N methods
    def nextstop(self,stop,no_of_neighbours):
        self.rcl_list=[]         
        unvisited2=self.unvisited[:]
        temp1=[]
        temp2=[]  
        for j in range(0,no_of_neighbours):
            for i in unvisited2:
                if self.distance_matrix[stop-1][i-1]!=0:
                    temp1.append(i)
                    temp2.append(self.distance_matrix[stop-1][i-1])
            s=min(temp2)
            position=temp2.index(s)
            self.rcl_list.append(temp1[position])
            unvisited2.remove(temp1[position])
            temp1=[]
            temp2=[]
        return self.rcl_list

    # This method selects a random city from the rcl_list to be put in the route.
    def random_choice(self,rcl_list):
        no=len(rcl_list)
        tempor=random.random()
        for i in range (0,no+1): 
            if tempor<=(i+1)/no:
               return rcl_list[i]
    
    # This method constructs a starting route by creating a list [1,2,...,N]
    # where N is the number of cities and then it returns the shuffled list.
    def construction_using_shuffle(self):
        temp_route=[]
        for i in range(1,self.N+1):
            temp_route.append(i)
        random.shuffle(temp_route)
        temp_route.append(temp_route[0])
        return temp_route
    
    
    # This method constructs a starting route by assigning a random city as the
    # starting city.At each iteration it picks the nearest neighbours (as many
    # as rcl_length is) and randomly select one out of them to put in the route
    # until there are no more cities to put in the route.
    def construction_using_GRASP_method(self,rcl_length):
        self.unvisited=[]
        for i in range(1,self.N+1):
            self.unvisited.append(i)
        route=[]
        first_stop=random.randint(1,self.N+1)
        route.append(first_stop)
        self.unvisited.remove(first_stop)
        for i in range(0,self.N-rcl_length):
            k = self.random_choice(self.nextstop(route[i],rcl_length))
            route.append(k)
            self.unvisited.remove(k)
        for i in range(self.N - rcl_length,self.N-1):                  
           self.nextstop((route[i-1]),len(self.unvisited))
           temp2=self.random_choice(self.rcl_list)
           route.append(temp2)
           position3=self.unvisited.index(temp2)
           del self.unvisited[position3]
        route.append(route[0])
        return route

    # Works exactly as the G.R.A.S.P method with and rcl_length = 1
    def construction_using_nearest_neighbour(self):
        self.unvisited = []
        for i in range(1, self.N+1):
            self.unvisited.append(i)
        route=[]
        first_stop=random.randint(1,self.N+1)
        route.append(first_stop)
        self.unvisited.remove(first_stop)
        for i in range(0,self.N-1):
            k=self.nextstop(route[i],1)
            route.append(k[0])
            self.unvisited.remove(k[0])
        route.append(route[0])
        return route

    def construction_using_double_nearest_neighbour(self):
        self.unvisited = []
        for i in range(1 , self.N+1):
            self.unvisited.append(i)
        route=[]
        first_stop=random.randint(1 , self.N+1)
        route.append(first_stop)
        self.unvisited.remove(first_stop)
        for i in range(0 , self.N-1):
            first_neighbour = self.nextstop(route[0],1)
            second_neighbour = self.nextstop(route[i],1)
            if self.distance_matrix[route[0]-1][first_neighbour[0]-1] < self.distance_matrix[route[i]-1][second_neighbour[0]-1]:
                route.insert(0,first_neighbour[0])
                self.unvisited.remove(first_neighbour[0])
            else:
                route.append(second_neighbour[0])
                self.unvisited.remove(second_neighbour[0])
        route.append(route[0])
        return route


##############################################################################
    

    def image(self,route,stops):
       temp1_route=[]
       for i in range(0,len(route)-1):
           temp1_route.append(stops[route[i]-1])
       temp1_route.append(temp1_route[0])
       m_val = [m[0] for m in temp1_route]
       n_val = [m[1] for m in temp1_route]
       plt.plot(m_val,n_val)
       plt.plot(m_val,n_val,'or')
       plt.plot(m_val[0],n_val[0],'ob')
       plt.show()

    # A method to calculate the total distance of a given route based on the 
    # inner distance matrix
    def total_route_distance(self):
        distance=0
        for i in range (0,len(self.route)-1):
            h = self.distance_matrix[self.route[i]-1][self.route[i+1]-1]
            distance += h
        return distance

    def best_route_distance(self):
        distance=0
        for i in range (0,len(self.best_route)-1):
            h = self.distance_matrix[self.best_route[i]-1][self.best_route[i+1]-1]
            distance += h
        return distance
######################################## VND PART###############################
    # The 2-opt operator that rearranges the inner subroute between two given
    # cities
    def two_opt_swap(self,start,end):    
        temp=self.route[start+1:end+1] 
        temp.reverse()
        for i in range(1,len(temp)):
            self.route[start+i]=temp[i]                                      
        return 

    # A simple swap operator 
    def swap(self,a,b):
        self.route[a],self.route[b] = self.route[b],self.route[a]
        return

    # This method searches and returns the  2-opt move that reduces the total
    # route distance the most
    def two_opt_neighbourhood_creation(self):
        tuples = []
        d=0
        for i in range(0,len(self.route)-2):
               for j in range(i+2,len(self.route)-1):
                   temp=self.distance_matrix[ self.route[i]-1][self.route[j]-1]+self.distance_matrix[self.route[i+1]-1][self.route[j+1]-1]-self.distance_matrix[self.route[i]-1][self.route[i+1]-1]-self.distance_matrix[self.route[j]-1][self.route[j+1]-1]
                   if temp<d:
                       d=temp
                       del tuples[:]    
                       tuples.append((i,j+1)) 
        return d,tuples  
    
    
    def two_opt(self):
        d,tuples = self.two_opt_neighbourhood_creation()
        if d<0:
            self.two_opt_swap(tuples[0][0],tuples[0][1])
            self.global_best = self.global_best + d
    
    # This method disturbs the current route in order to escape from local
    # minima.
    
    def shake(self,mode):
        if mode==1:
            temp1 = random.randint(1,self.N-2)
            temp2 = random.randint(1,self.N-2)
            if temp1 != self.route[0]:
                if temp1!= self.route[self.N]:
                    if temp2 != self.route[0]:
                        if temp2!= self.route[self.N]:
                             self.two_opt_swap(self.route[temp1],self.route[temp2])
        if mode==2:
            temp1 = random.randint(1,self.N-1)
            temp2 = random.randint(1,self.N-1)
            if temp1 != self.route[0]:
                if temp1!= self.route[self.N]:
                    if temp2 != self.route[0]:
                        if temp2!= self.route[self.N]:
                            self.swap(temp1,temp2)    
        return 
    
    
    # The VND meta-heuristic method 
    # *** Note to self : add more neighbourhoods , and operators
    def descend(self):
        old_best = self.global_best+1
    
        while self.global_best<old_best:
            old_best = self.global_best
            self.two_opt()
            print(self.total_route_distance())
        if self.best_route_distance() < self.total_route_distance():
            self.route = self.best_route
        else :
            self.best_route = self.route
        return 

if __name__ == '__main__':

    url = 'http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/a280.tsp'
    filepath = input("Give Filepath:")
    # Uncomment the line below for a test run
    # filepath = url
    obj = VND(str(filepath))
    iterations = input('Number of iteration for the VND method:')

             
    for i in range (int(iterations)):
        print(obj.best_route)
        obj.image(obj.best_route,obj.stops)
        obj.shake(mode=2)
        print(obj.best_route)
        obj.image(obj.best_route,obj.stops)
        obj.descend()
        print(obj.best_route)
        obj.image(obj.best_route,obj.stops)

    print(obj.best_route)
    print(obj.best_route_distance())
    obj.image(obj.best_route,obj.stops)        
        
