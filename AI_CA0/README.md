<h1 style = "font-size: 30px; text-align: center;">AI Genetic Hands On</h1>
<h2 style = "font-size: 25px; text-align: center;">Hospital Job Scheduling</h2>
<h2 style = "font-size: 25px; text-align: center; color: #666">Name: Sepehr Ghobadi</h2>
<h2 style = "font-size: 25px; text-align: center; color: #666">Student Id: 810098009</h2>
<h4 style="text-align: center">Spring 1400</h4>

# Goal Of Project

In this project, we apply the Genetic Algorithms to the problem of hospital job scheduling. There are N working days and each day is partitioned in three shifts: morning, evening and night shifts . There is constraints on each shift on minimum and maximum number of doctors that are available in the shift. Also a doctor cant do night shifts for 3 consecutive nights and also cant do morning or evening shift the day after he or she is doing night shift. There is a limit for a single doctor's number of shifts in N days. Since the Genetic Algorithm is used for efficiently searching a large problem space, it can find the solution to the scheduling problem in a reasonable time. The methodology and codes are all explained throughout this report and the effect of several related parameters and techniques are discussed.

# 1. Introducing the Genetic Algorithm Related Concepts in the Problem

## 1.1 Genes And Chromosomes

Chromosomes are defined by __N__ binary segments. Each segment represents the schedule of that day. Each day's segment containts three __D__ (number of doctors) bit binary string representing doctors status in morning, evening and nights shifts in that order. if __i__'th bit of a shift is __0__ means that __i__'th doctor is resting that shift and if its __1__ the __i__'th doctor is working in that shift

there is no exact definition oof genes in this model. as we do crossover based on days we can consider each days string as genes. also we do mutation on single bits so bits can be genes too.

<h4><center> Chromosome = (Gene of day 1 | Gene of day 2 | .... | Gene of day N) </center></h4>
<h4><center> Gene of day i = (status of morning shift | status of evening shift | status of night shift) </center></h4>
<h4><center> Status of shift j = (presence of doctor 1 | presence of doctor 2 | ....| presence of doctor D) </center></h4>

## 1.2 Crossover

In the Genetic Algorithm, Crossover is done to combine the genetic information of two parents to generate new offspring. The regular method of crossover (simple K point or uniform crossover) can be used with this model as there is no dependency between chromosome's bits. We use simple K point crossover. 

## 1.3 Mutation

Also mutation with this model is straight forward. As there is no dependency between bits a single bit cant be mutated independently of other bits.
Mutating a bit means changing a single doctor's presence on a shift

The chance of occurring mutation in each chromosome at each generation is represented by Mutation Probability. This number can be a constant or can get reduced for a specific amount after each generation. Reducing Strategy is due to the fact that it is expected that as more generations are produced, the need for random decisions will decrease and searching the problem's whole space is going to reduce. In this problem we use simple constant probability for mutation

## 1.4 Fitness Function

Performance of a schedule is measured by two factors. First one is number of shifts that dont violate constraints on number of doctors working in that shift. or each shift that vioilates this constraints __ShiftsCost__ increases by one.

Second one is number of doctors that violate constraints on night shifts or maximum working capacity of a single doctor. Each doctor can violates multiple rules in a single scheduling (for example working more than 3 consecutive nights and also working more than limit in total). For each violation __DoctorsCost__ increases by one.

The total cost function is calculated by bellow formula:
<h4><center> Total Cost = DoctorsPenaltyWeight*DoctorsCost + ShiftsPenaltyWeight*ShiftsCost </center></h4>

In this problem __DoctorsPenaltyWeight__ and __ShiftsPenaltyWeight__ are equal but different weightings can be used when priorities and goals differ.

## 1.5 Choosing Chromosomes for Population Refinement

### 1.5.1 Selecting the Remaining Chromosomes

After evaluating the population in each generation using the fitness function, chromosomes with better performance must remain in the next generation while others must be replaced by a new generation. While selecting chromosomes with the highest fitness function values as the survived population seems to be a reasonable procedure, in action it is observed that giving small random chances of survival to other chromosomes with lower performance will sometimes improve the efficiency of the algorithm. In this project a certain percent of best chromosomes will be transfered too next generation directly(with out crossover and mating). From rest of chromosomes selecting The survived population is based on a random procedure in which each of the chromosomes will receive a probability for survival and the new population is selected based on the mentioned probabilities form the population. The more the survival probability of a chromosome be the more its chance for survival will be.

The survival probability for each of the chromosomes are computed using the below formula:

<h2>$$𝑃_𝑖=\frac{𝑒^{-𝛼𝑉_𝑖}}{\Sigma_{n=1}^{N}𝑒^{-𝛼𝑉_𝑛}}$$</h2>
    
where __P_i__ is the survival probability,  __𝑉_𝑖__  is the fitness value of the __𝑖__'th chromosome, __𝑁__ is the number of chromosomes and __𝛼__ is a non-negative small number. For  __𝛼__=0 , the probability of survival would be equal for all of the chromosomes in the population and as __𝛼__ grows, the probability of survival for chromosomes with higher fitness values (in this problem lower fitness values are equal to negative cost of a schedule so they are considered better) will decrease. Exponential terms are used for computing the probabilities in order to reinforce the difference between the chance of survival of chromosomes with higher and lower fitness function values.

### 1.5.2 Choosing Couples for Crossover Operation

The chromosome couples for the crossover operation are selected from the selected chromosomes to be remain. The process of choosing the parents is also in a random procedure. We choose random couples from chromosomes and perform crossoover on them with a constant probability

Other algorithms can be used for mating of chromosomes. a simple algortihm could use product of two chromosomes' fitness value as probabilty of mating those two. The bigger the product of the values of two chromosomes, the higher their chance of being selected for crossover.


<h2>$$𝑄_{𝑖,𝑗} = \frac{𝑒^{𝛽𝑉_𝑖𝑉_𝑗}}{ \Sigma_{n=1}^{N}{\Sigma_{m=n+1}^{N} 𝑒^{𝛽𝑉_𝑛𝑉_𝑚}}} , 𝑖>𝑗 $$</h2>
 
where __Q_i_j__  is the probability, __V_i__  is the fitness value of the __i__ 'th chromosome, __V_j__  is the fitness value of the __j__'th chromosome, __N__ is the number of chromosomes from which we are selecting and __𝛽__  is an anon-negative small number. For  __𝛽__=0 , the probability of selection would be equal for each chromosome couples in the population and as __𝛽__ grows, the probability of being chosen for chromosomes with a lower product of fitness values will decrease. Like the formula mentioned in the previous section, exponential terms are used for computing the probabilities in order to reinforce the difference between the chance of crossover for more eligible couples.



```python
import random
import math
import numpy as np
```

<h2 style = "font-size: 25px;">Test Files</h2>


```python
testFile1 = "test1.txt"
testFile2 = "test2.txt"
```

<h2 style = "font-size: 25px;">Reading Test File Content</h2>


```python
def readInput(testFile) :
    file = open(testFile, 'r+')
    fileList = file.readlines()
    fileList = [s.replace('\n', '') for s in fileList]
    
    [days, doctors] = [int(i) for i in fileList[0].split()]
    maxCapacity = int(fileList[1])
    
    allShifts = []
    for i in range(2, days + 2):
        dayRequirements = fileList[i].split()
        morningReqs = [int(i) for i in dayRequirements[0].split(",")]
        eveningReqs = [int(i) for i in dayRequirements[1].split(",")]
        nightReqs = [int(i) for i in dayRequirements[2].split(",")]
        
        allShifts.append((morningReqs, eveningReqs, nightReqs))

    file.close()
    return [days, list(range(doctors)), maxCapacity, allShifts]
```

<h2 style = "font-size: 25px;">Job Scheduler</h2>


```python
class JobScheduler:
    def __init__(self, fileInfo):
        self.days = fileInfo[0]
        self.doctors = len(fileInfo[1])
        self.doctorsIds = fileInfo[1]
        self.maxCapacity = fileInfo[2]
        self.allShifts = fileInfo[3]
        self.popSize = 300
        self.crossOverPoints = [int(i*(self.days/8)) for i in range(9)]
        self.elitismPercentage = 0
        self.pc = 1
        self.pm = 0.01
        self.population = self.generateInitialPopulation()
        
        
    def generateInitialPopulation(self):
        population = []
        for i in range(self.popSize):
            chromosome = ""
            randInt = random.randint(0, 2**(3*self.days*self.doctors)-1 )
            for b in range(3*self.days*self.doctors) :
                if randInt & (1 << b):
                    chromosome += "1"
                else:
                    chromosome += "0"
            population.append( (chromosome, self.calculateFitness(chromosome)) )
        return population
        
    
    def crossOver(self, chromosome1, chromosome2):
        newchromosome1 = ""
        newchromosome2 = ""
        for i in range(1,len(self.crossOverPoints)):
            if i%2 == 0 :
                newchromosome1 += chromosome1[3*self.crossOverPoints[i-1]*self.doctors:3*self.crossOverPoints[i]*self.doctors]
                newchromosome2 += chromosome2[3*self.crossOverPoints[i-1]*self.doctors:3*self.crossOverPoints[i]*self.doctors]
            else:
                newchromosome1 += chromosome2[3*self.crossOverPoints[i-1]*self.doctors:3*self.crossOverPoints[i]*self.doctors]
                newchromosome2 += chromosome1[3*self.crossOverPoints[i-1]*self.doctors:3*self.crossOverPoints[i]*self.doctors]
        return (newchromosome1, newchromosome2)
        
                
    def mutate(self, chromosome):
        newChromosome = ""
        for c in chromosome:
            rand = random.random()
            if rand < self.pm :
                newChromosome += str( 1 - int(c) )
            else:
                newChromosome += c
        return newChromosome
    
    def getShiftsData(self, chromosome):
        daysData = [ chromosome[(3*i*self.doctors) : (3*(i+1)*self.doctors) ] for i in range(self.days) ]
        shiftsData = []
        for d in daysData:
            shiftsData.append( [ d[j*self.doctors : (j+1)*self.doctors] for j in range(3) ] )
        return shiftsData
        
    def calculateFitness(self, chromosome):
        doctorsWeight = 1
        doctorsCost = 0
        shiftsWeight = 1
        shiftsCost = 0
        shiftsData = self.getShiftsData(chromosome)
        for day in range(self.days):
            for shift in range(3):
                doctorsCount = shiftsData[day][shift].count("1")
                if (doctorsCount < self.allShifts[day][shift][0] ) or (self.allShifts[day][shift][1] < doctorsCount) :
                    shiftsCost += 1
            if day>0:
                for doctor in range(self.doctors):
                    if shiftsData[day-1][2][doctor] == "1" and ( shiftsData[day][0][doctor] == "1" or shiftsData[day][1][doctor] == "1" ):
                        doctorsCost += 1
                    if day < self.days-1 and shiftsData[day-1][2][doctor] == "1" and shiftsData[day][2][doctor] == "1" and shiftsData[day+1][2][doctor] == "1" :
                        doctorsCost += 1
        for doctor in range(self.doctors):
            workDays = 0;
            for day in range(self.days):
                for shift in range(3):
                    if shiftsData[day][shift][doctor] == "1":
                        workDays += 1
            if workDays > self.maxCapacity:
                doctorsCost += 1
        totalCost = doctorsWeight*doctorsCost + shiftsWeight*shiftsCost
        return totalCost
    
    
    def generateNewPopulation(self):
        alpha = 1
        self.population = sorted(self.population, key=lambda p: p[1])
        bestChromosomes = [ p[0] for p in self.population[0: int((self.elitismPercentage/100)*self.popSize)] ]
        remainedChromosomes = [ p[0] for p in self.population[int((self.elitismPercentage/100)*self.popSize):] ]
        fitnessSum = sum([ math.exp(-1*alpha*p[1]) for p in self.population[int((self.elitismPercentage/100)*self.popSize):] ])
        surviveProbs = [ (math.exp(-1*alpha*p[1])/fitnessSum) for p in self.population[int((self.elitismPercentage/100)*self.popSize):] ]
        survivedChromosoms = np.random.choice( remainedChromosomes, len(remainedChromosomes), replace=True, p=surviveProbs).tolist()
        random.shuffle(survivedChromosoms)
        newChromosomes = bestChromosomes
        for i in range(0,len(survivedChromosoms),2):
            if i+1 >= len(survivedChromosoms) :
                newChromosomes.append(survivedChromosoms[i])
            else:
                rand = random.random()
                chromosome1 = survivedChromosoms[i]
                chromosome2 = survivedChromosoms[i+1]
                if rand < self.pc :
                    chromosome1, chromosome2 = self.crossOver(chromosome1, chromosome2)
                newChromosomes.append(chromosome1)
                newChromosomes.append(chromosome2)
        self.population = []
        for chromosome in newChromosomes:
            chromosome = self.mutate(chromosome)
            self.population.append( (chromosome, self.calculateFitness(chromosome)) )
        return
    
    def print_schedule(self, chromosome, file):
        shiftsData = self.getShiftsData(chromosome)
        schedule = ""
        for day in range(self.days):
            result = ""
            for shift in range(3):
                first = True
                for doctor in range(self.doctors):
                    if shiftsData[day][shift][doctor] == "1":
                        if first == False:
                            result += ","
                        first = False
                        result += str(self.doctorsIds[doctor])
                if shift<2:
                    result += " "
            schedule += result + "\n"
        f = open(file, "w")
        f.write(schedule)
        f.close()
                    
            
                
    
    def check_finish(self):
        result = {"min_cost":100000000000000000000, "chromosome":None}
        for chromosome, fitness in self.population:
            result["min_cost"] = min(result["min_cost"], fitness)
            if fitness == 0 :
                result["chromosome"]=chromosome
                return result
        return result
    
    def schedule(self, log=False):
        iteration = 1
        result = self.check_finish()
        while (result["min_cost"] > 0 ):
            if log:
                print("iteration {}\t\t best chromosome fitness: {}".format(iteration, result["min_cost"]))
            self.generateNewPopulation()
            result = self.check_finish()
            iteration += 1
        if log:
            print("Finished !")
        return result
```

<h2 style = "font-size: 25px;">Execution</h2>


```python
import time

fileInfo1 = readInput(testFile1)

RUNS = 3
meanTime=0

for run in range(RUNS):

    start = time.time()

    scheduler = JobScheduler(fileInfo1)
    result = scheduler.schedule(log=(run==0))
    scheduler.print_schedule(result["chromosome"], "output1.txt")

    end = time.time()
    meanTime += (end-start)

print("test 1: ", '%.2f'%(meanTime/RUNS), 'sec,', RUNS, 'runs')
```

    iteration 1		 best chromosome fitness: 40
    iteration 2		 best chromosome fitness: 35
    iteration 3		 best chromosome fitness: 32
    iteration 4		 best chromosome fitness: 29
    iteration 5		 best chromosome fitness: 26
    iteration 6		 best chromosome fitness: 25
    iteration 7		 best chromosome fitness: 22
    iteration 8		 best chromosome fitness: 20
    iteration 9		 best chromosome fitness: 17
    iteration 10		 best chromosome fitness: 16
    iteration 11		 best chromosome fitness: 15
    iteration 12		 best chromosome fitness: 15
    iteration 13		 best chromosome fitness: 14
    iteration 14		 best chromosome fitness: 14
    iteration 15		 best chromosome fitness: 13
    iteration 16		 best chromosome fitness: 12
    iteration 17		 best chromosome fitness: 12
    iteration 18		 best chromosome fitness: 10
    iteration 19		 best chromosome fitness: 9
    iteration 20		 best chromosome fitness: 9
    iteration 21		 best chromosome fitness: 9
    iteration 22		 best chromosome fitness: 8
    iteration 23		 best chromosome fitness: 8
    iteration 24		 best chromosome fitness: 8
    iteration 25		 best chromosome fitness: 7
    iteration 26		 best chromosome fitness: 7
    iteration 27		 best chromosome fitness: 6
    iteration 28		 best chromosome fitness: 6
    iteration 29		 best chromosome fitness: 6
    iteration 30		 best chromosome fitness: 6
    iteration 31		 best chromosome fitness: 5
    iteration 32		 best chromosome fitness: 4
    iteration 33		 best chromosome fitness: 4
    iteration 34		 best chromosome fitness: 4
    iteration 35		 best chromosome fitness: 3
    iteration 36		 best chromosome fitness: 3
    iteration 37		 best chromosome fitness: 2
    iteration 38		 best chromosome fitness: 2
    iteration 39		 best chromosome fitness: 2
    iteration 40		 best chromosome fitness: 2
    iteration 41		 best chromosome fitness: 2
    iteration 42		 best chromosome fitness: 2
    iteration 43		 best chromosome fitness: 2
    iteration 44		 best chromosome fitness: 2
    iteration 45		 best chromosome fitness: 2
    iteration 46		 best chromosome fitness: 2
    iteration 47		 best chromosome fitness: 2
    iteration 48		 best chromosome fitness: 1
    iteration 49		 best chromosome fitness: 1
    iteration 50		 best chromosome fitness: 1
    iteration 51		 best chromosome fitness: 1
    iteration 52		 best chromosome fitness: 1
    iteration 53		 best chromosome fitness: 1
    iteration 54		 best chromosome fitness: 1
    iteration 55		 best chromosome fitness: 1
    iteration 56		 best chromosome fitness: 1
    iteration 57		 best chromosome fitness: 1
    iteration 58		 best chromosome fitness: 1
    iteration 59		 best chromosome fitness: 1
    iteration 60		 best chromosome fitness: 1
    iteration 61		 best chromosome fitness: 1
    iteration 62		 best chromosome fitness: 1
    iteration 63		 best chromosome fitness: 1
    iteration 64		 best chromosome fitness: 1
    iteration 65		 best chromosome fitness: 1
    iteration 66		 best chromosome fitness: 1
    Finished !
    test 1:  1.67 sec, 3 runs



```python
fileInfo2 = readInput(testFile2)

RUNS = 3
meanTime=0

for run in range(RUNS):

    start = time.time()

    scheduler = JobScheduler(fileInfo2)
    result = scheduler.schedule(log=(run==0))
    scheduler.print_schedule(result["chromosome"], "output2.txt")

    end = time.time()
    meanTime += (end-start)

print("test 2: ", '%.2f'%(meanTime/RUNS), 'sec,', RUNS, 'runs')
```

    iteration 1		 best chromosome fitness: 79
    iteration 2		 best chromosome fitness: 75
    iteration 3		 best chromosome fitness: 71
    iteration 4		 best chromosome fitness: 68
    iteration 5		 best chromosome fitness: 63
    iteration 6		 best chromosome fitness: 61
    iteration 7		 best chromosome fitness: 58
    iteration 8		 best chromosome fitness: 54
    iteration 9		 best chromosome fitness: 49
    iteration 10		 best chromosome fitness: 46
    iteration 11		 best chromosome fitness: 44
    iteration 12		 best chromosome fitness: 41
    iteration 13		 best chromosome fitness: 40
    iteration 14		 best chromosome fitness: 37
    iteration 15		 best chromosome fitness: 35
    iteration 16		 best chromosome fitness: 33
    iteration 17		 best chromosome fitness: 32
    iteration 18		 best chromosome fitness: 31
    iteration 19		 best chromosome fitness: 29
    iteration 20		 best chromosome fitness: 28
    iteration 21		 best chromosome fitness: 27
    iteration 22		 best chromosome fitness: 26
    iteration 23		 best chromosome fitness: 25
    iteration 24		 best chromosome fitness: 23
    iteration 25		 best chromosome fitness: 22
    iteration 26		 best chromosome fitness: 22
    iteration 27		 best chromosome fitness: 21
    iteration 28		 best chromosome fitness: 21
    iteration 29		 best chromosome fitness: 20
    iteration 30		 best chromosome fitness: 19
    iteration 31		 best chromosome fitness: 19
    iteration 32		 best chromosome fitness: 17
    iteration 33		 best chromosome fitness: 17
    iteration 34		 best chromosome fitness: 16
    iteration 35		 best chromosome fitness: 16
    iteration 36		 best chromosome fitness: 16
    iteration 37		 best chromosome fitness: 15
    iteration 38		 best chromosome fitness: 15
    iteration 39		 best chromosome fitness: 15
    iteration 40		 best chromosome fitness: 15
    iteration 41		 best chromosome fitness: 15
    iteration 42		 best chromosome fitness: 15
    iteration 43		 best chromosome fitness: 15
    iteration 44		 best chromosome fitness: 14
    iteration 45		 best chromosome fitness: 14
    iteration 46		 best chromosome fitness: 12
    iteration 47		 best chromosome fitness: 12
    iteration 48		 best chromosome fitness: 12
    iteration 49		 best chromosome fitness: 12
    iteration 50		 best chromosome fitness: 12
    iteration 51		 best chromosome fitness: 12
    iteration 52		 best chromosome fitness: 12
    iteration 53		 best chromosome fitness: 12
    iteration 54		 best chromosome fitness: 12
    iteration 55		 best chromosome fitness: 12
    iteration 56		 best chromosome fitness: 12
    iteration 57		 best chromosome fitness: 11
    iteration 58		 best chromosome fitness: 10
    iteration 59		 best chromosome fitness: 10
    iteration 60		 best chromosome fitness: 10
    iteration 61		 best chromosome fitness: 10
    iteration 62		 best chromosome fitness: 10
    iteration 63		 best chromosome fitness: 10
    iteration 64		 best chromosome fitness: 10
    iteration 65		 best chromosome fitness: 10
    iteration 66		 best chromosome fitness: 10
    iteration 67		 best chromosome fitness: 9
    iteration 68		 best chromosome fitness: 8
    iteration 69		 best chromosome fitness: 8
    iteration 70		 best chromosome fitness: 8
    iteration 71		 best chromosome fitness: 7
    iteration 72		 best chromosome fitness: 7
    iteration 73		 best chromosome fitness: 7
    iteration 74		 best chromosome fitness: 7
    iteration 75		 best chromosome fitness: 7
    iteration 76		 best chromosome fitness: 7
    iteration 77		 best chromosome fitness: 7
    iteration 78		 best chromosome fitness: 7
    iteration 79		 best chromosome fitness: 7
    iteration 80		 best chromosome fitness: 7
    iteration 81		 best chromosome fitness: 6
    iteration 82		 best chromosome fitness: 6
    iteration 83		 best chromosome fitness: 6
    iteration 84		 best chromosome fitness: 6
    iteration 85		 best chromosome fitness: 6
    iteration 86		 best chromosome fitness: 6
    iteration 87		 best chromosome fitness: 6
    iteration 88		 best chromosome fitness: 6
    iteration 89		 best chromosome fitness: 6
    iteration 90		 best chromosome fitness: 6
    iteration 91		 best chromosome fitness: 5
    iteration 92		 best chromosome fitness: 5
    iteration 93		 best chromosome fitness: 5
    iteration 94		 best chromosome fitness: 5
    iteration 95		 best chromosome fitness: 5
    iteration 96		 best chromosome fitness: 5
    iteration 97		 best chromosome fitness: 5
    iteration 98		 best chromosome fitness: 5
    iteration 99		 best chromosome fitness: 5
    iteration 100		 best chromosome fitness: 5
    iteration 101		 best chromosome fitness: 5
    iteration 102		 best chromosome fitness: 5
    iteration 103		 best chromosome fitness: 5
    iteration 104		 best chromosome fitness: 5
    iteration 105		 best chromosome fitness: 5
    iteration 106		 best chromosome fitness: 5
    iteration 107		 best chromosome fitness: 5
    iteration 108		 best chromosome fitness: 5
    iteration 109		 best chromosome fitness: 5
    iteration 110		 best chromosome fitness: 5
    iteration 111		 best chromosome fitness: 5
    iteration 112		 best chromosome fitness: 5
    iteration 113		 best chromosome fitness: 5
    iteration 114		 best chromosome fitness: 5
    iteration 115		 best chromosome fitness: 5
    iteration 116		 best chromosome fitness: 5
    iteration 117		 best chromosome fitness: 5
    iteration 118		 best chromosome fitness: 5
    iteration 119		 best chromosome fitness: 5
    iteration 120		 best chromosome fitness: 5
    iteration 121		 best chromosome fitness: 5
    iteration 122		 best chromosome fitness: 5
    iteration 123		 best chromosome fitness: 5
    iteration 124		 best chromosome fitness: 5
    iteration 125		 best chromosome fitness: 5
    iteration 126		 best chromosome fitness: 4
    iteration 127		 best chromosome fitness: 4
    iteration 128		 best chromosome fitness: 4
    iteration 129		 best chromosome fitness: 4
    iteration 130		 best chromosome fitness: 4
    iteration 131		 best chromosome fitness: 4
    iteration 132		 best chromosome fitness: 4
    iteration 133		 best chromosome fitness: 4
    iteration 134		 best chromosome fitness: 4
    iteration 135		 best chromosome fitness: 4
    iteration 136		 best chromosome fitness: 4
    iteration 137		 best chromosome fitness: 4
    iteration 138		 best chromosome fitness: 4
    iteration 139		 best chromosome fitness: 4
    iteration 140		 best chromosome fitness: 4
    iteration 141		 best chromosome fitness: 4
    iteration 142		 best chromosome fitness: 4
    iteration 143		 best chromosome fitness: 5
    iteration 144		 best chromosome fitness: 4
    iteration 145		 best chromosome fitness: 4
    iteration 146		 best chromosome fitness: 3
    iteration 147		 best chromosome fitness: 3
    iteration 148		 best chromosome fitness: 3
    iteration 149		 best chromosome fitness: 3
    iteration 150		 best chromosome fitness: 3
    iteration 151		 best chromosome fitness: 3
    iteration 152		 best chromosome fitness: 3
    iteration 153		 best chromosome fitness: 3
    iteration 154		 best chromosome fitness: 3
    iteration 155		 best chromosome fitness: 3
    iteration 156		 best chromosome fitness: 3
    iteration 157		 best chromosome fitness: 3
    iteration 158		 best chromosome fitness: 2
    iteration 159		 best chromosome fitness: 2
    iteration 160		 best chromosome fitness: 2
    iteration 161		 best chromosome fitness: 2
    iteration 162		 best chromosome fitness: 2
    iteration 163		 best chromosome fitness: 2
    iteration 164		 best chromosome fitness: 2
    iteration 165		 best chromosome fitness: 2
    iteration 166		 best chromosome fitness: 2
    iteration 167		 best chromosome fitness: 2
    iteration 168		 best chromosome fitness: 2
    iteration 169		 best chromosome fitness: 2
    iteration 170		 best chromosome fitness: 2
    iteration 171		 best chromosome fitness: 1
    iteration 172		 best chromosome fitness: 1
    iteration 173		 best chromosome fitness: 1
    iteration 174		 best chromosome fitness: 1
    iteration 175		 best chromosome fitness: 1
    iteration 176		 best chromosome fitness: 1
    iteration 177		 best chromosome fitness: 1
    iteration 178		 best chromosome fitness: 1
    iteration 179		 best chromosome fitness: 1
    iteration 180		 best chromosome fitness: 1
    iteration 181		 best chromosome fitness: 1
    iteration 182		 best chromosome fitness: 1
    iteration 183		 best chromosome fitness: 1
    Finished !
    test 2:  13.99 sec, 3 runs


# 2 Discussing the Results and Effect of Parameters

## 2.1 Mutation Vs. Crossover

Generally speaking, while Crossover accounts for pulling the population towards a local minimum/maximum, Mutation is intended to occasionally break one or more members of a population out of a local minimum/maximum space and potentially discover a better minimum/maximum space. In other words, __Crossover is a convergence operation and responsible for fast convergence__ while __Mutation is a divergence operation avoiding the population to become too similar to each other__.

Since the end goal is to bring the population to convergence, usually crossover happens more frequently. Mutation should happen less frequently to bring diversity into the population and typically only affects a few members of a population in any given generation.

This does not mean that Crossover is more important than Mutation or that Mutation must not be used at all. __Both of the mentioned operations are important for adequate convergence and must be applied at their place__.

__If the mutation operation is not applied in the algorithm__, the similarity between chromosomes would occur after few generations, meaning the algorithm has stuck into local minimum/maximum and would become incapable of finding the global optimal solution.

__If the crossover operation is not applied to the algorithm__, fast and adequate convergence is not possible at all and the process would become a simple exhaustive search.

## 2.2 The Effect of The Population Size

The population size plays a key role in the Genetic Algorithm's performance and it is important to be set properly. For small population size, the Genetic Algorithm can __converge early on a slope and it will not provide an adequate initial surface exploration__. But __this fact does not necessarily mean that a larger population size always ends with better results__. A large population __takes an excessive number of function evaluations__ to converge and __slow down the process of finding the optimal solution__.

## 2.3 Avoid Getting Stuck on Local Optimum

Like other evolutionary algorithms, the Genetic Algorithm can not always guarantee to reach the best answer and its performance reduced by falling in local minimum/maximums. While there is no certain method for avoiding such situations, the following approaches may be found useful:

   &emsp;- Multiple runs with different seeds of initial chromosomes are the easiest way to find the optimal solution            to a problem and can sometimes preserve the algorithm from getting stuck in local optimums (known as Random Restart Algorithm).


   &emsp;- Increasing the mutation probability for each generation while running the algorithm may remedy the problem.            As it was discussed earlier, the main purpose of using the Mutation technique is to avoid getting stuck in            local optimums and exploring more of the problem space. Hence, increasing the Mutation operations while              generating a new population, seems a logical way out of the problem.

&emsp;- Increasing the population which is kept at each generation of the algorithm may also help to escape from local optimums since larger population would explore more of the problem space which lowers the chance of falling in local minimums/maximums (but also deacreases the speed).

&emsp;- It is observed that giving a small probability of survival and attending crossover to chromosomes with lower fitness function values can sometimes preserve the Genetic algorithm from local optimums.


Some of these techniques mentioned above were used in the project to avoid local optimums and improve the performance and speed of the genetic algorithm and some others were unnecessary in this problem.


```python

```
