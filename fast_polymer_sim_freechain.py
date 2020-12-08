'''
Developer: Pallav Chanda <pallav16@iiserb.ac.in>; 16131, BSMS 2016; IISER Bhopal
Month: December 2019 (Start Date: 07.12.19)
Code written during a short-term project under Prof. Snigdha Thakur <snigdha@iiserb.ac.in>, IISER Bhopal
Code based on:

'''

import numpy as np
import random
import array
import itertools
import matplotlib.pyplot as plt

####---------------AIM OF THE CODE---------------####

print('\n\nThis code simulates looping events for Free-Chain Polymers.\nReturns the average number of MCS (Monte Carlo Steps) the algorithm takes till interaction.\n')

####------------------VARIABLES------------------####

l = 1.
N = int(input('Enter number of monomers in the polymer: ')) #chain length (number of monmers) of the polymer
Nm1 = N - 1
eq_period = 0 #Equilibration period

#Getting Equilibration period
Ns = np.array([8, 16, 24, 32, 40, 48, 56, 64, 128, 256])
periods = 1000* np.array([5, 20, 50, 100, 150, 250, 300, 400, 1600, 6400])
for i in range(len(Ns)):
    if (Ns[i] == N): eq_period = periods[i] * N
if N not in Ns:
    print('Set equilibration period manually!!!')
    eq_period = int(input('Enter equilibration period: '))

#Counters and Adders        
TnMCS = 0.
arrnMCS = []
max_iter = int(input('Enter the number of looping events you want to simulate: '))
max_trials = 80000000 #Prevent infinite looping

#Positions of interacting monomers
im1, im2 = int(input('\nEnter position of 1st interacting monomer: ') - 1), int(input('Enter position of 2nd interacting monomer: ') - 1)
print('\nYou want: Interaction between %d-th and %d-th monomers.\n' % (im1 + 1, im2 + 1))

#Bond-Fluctuation Condition
'''NOTE: We hardly use this! We use the 'allowed Bond Vectors set' for comparing.'''
blmin = 2.*l #minimum bond-length
blmax = np.sqrt(10)*l #maximum bond-length

#Set of allowed Bond Vectors, we have to include all permutations (including +, -)
'''NOTE: Please see "J. Chem. Phys. 94, 2294 (1991); https://doi.org/10.1063/1.459901" for more details regarding this set!'''
'''We define the set such that elements of each individual allowed Bond Vector is in ascending order.
While comparing with this set later, we will take absolute value of the bond vectors after the Monte Carlo move
and arrange their elements in ascending order before comparing.
This will eliminate the need for defining all possible permutations of individual allowed Bond Vectors in Bset.'''
Bset = [[0., 0., 2.], [0., 1., 2.], [1., 1., 2.], [1., 2., 2.], [0., 0., 3.], [0., 1., 3.]]

#Direction arrays for choosing random direction to move
dirxyz = [0, 1, 2] #direction to move in. 0=x, 1=y, 2=z.
dirud = [1, -1] #direction to move after choozing x/y/z. +1=+ve, -1=-ve

####-------------FUNCTION DEFINITIONS------------####

#Function to chcek SAW condition (Each lattice site can be occupied by only one monomer)
def SAWcond(k_, rx_, ry_, rz_, nrx_, nry_, nrz_):
    satisfied = 1
    for j in range(N):
        if (j != k_ and abs(rx_[j] - nrx_) < 2.*l and abs(ry_[j] - nry_) < 2.*l and abs(rz_[j] - nrz_) < 2.*l):
            satisfied = 0
            break
    return satisfied

#Function to chcek Bond-length condition along with no bond-crossing
def BLC(k_, rx_, ry_, rz_, nrx_, nry_, nrz_):
    inset = 0
    if (k_ == 0):
        bvec2 = [abs(rx_[k+1] - nrx_), abs(ry_[k+1] - nry_), abs(rz_[k+1] - nrz_)]
        bvec2.sort()
        if bvec2 in Bset: inset = 1
    elif (k == N-1):
        bvec1 = [abs(nrx_ - rx_[k-1]), abs(nry_ - ry_[k-1]), abs(nrz_ - rz_[k-1])]
        bvec1.sort()
        if bvec1 in Bset: inset = 1
    else:
        bvec1, bvec2 = [abs(nrx_ - rx_[k-1]), abs(nry_ - ry_[k-1]), abs(nrz_ - rz_[k-1])], \
        [abs(rx_[k+1] - nrx_), abs(ry_[k+1] - nry_), abs(rz_[k+1] - nrz_)]
        bvec1.sort()
        bvec2.sort()
        if ((bvec1 in Bset) and (bvec2 in Bset)): inset = 1
    return inset

####------------------MAIN CODE------------------####

print('Finding average interaction time for %d different modeled polymers.\n' % (max_iter))

for iteration in range(max_iter):
    print('Simulating looping event:', iteration+1)
    
    '''####----Variables and Arrays that need to be RESET after each run----####'''

    #Counters, flags
    neaccept = 0 #Accepted values during equilibration period
    npaccept = 0 #Accepted values till the interacting monomers come close
    trials = 0

    #Arrays for storing positions of monomers in a polymer
    rx = [0.] * N #The polymer #r[monomer]
    ry = [0.] * N
    rz = [0.] * N
    
    #Generating initial modeled chain
    i = 1
    while i < N:
        rx[i], ry[i], rz[i] = rx[i-1], ry[i-1], rz[i-1]
        dxyz, pm = random.choice(dirxyz), random.choice(dirud)
        if (dxyz == 0): rx[i] += pm * 2.*l
        elif (dxyz ==1): ry[i] += pm * 2.*l
        else: rz[i] += pm * 2.*l
        for j in range(i):
            if (rx[i] == rx[j] and ry[i] == ry[j] and rz[i] == rz[j]):
                i -= 1
                break
        i += 1

    #Initializing arrays
    ivec = [] #Interaction vector to check if the interacting monomers came close
    
    '''####----Algorithm implementation starts here----####'''
    
    '''Monte Carlo Equilibration of Modeled Chain'''
    
    randdirs = random.choices(dirxyz, k = eq_period)
    randpms = random.choices(dirud, k = eq_period)
    
    for i in range(eq_period):
        k = random.randint(0, Nm1) #Choose k-th atom for displacement
        #k = randks[i]
        dxyz, pm = randdirs[i], randpms[i]
        #dxyz, pm = random.choice(dirxyz), random.choice(dirud)
        nrx, nry, nrz = rx[k], ry[k], rz[k]
        if (dxyz == 0): nrx += pm * l
        elif (dxyz == 1): nry += pm * l
        else: nrz += pm * l
    
        #Bond-length condition along with no bond-crossing
        BLCcheck = BLC(k, rx, ry, rz, nrx, nry, nrz)
        if (BLCcheck == 0): continue
        
        #SAW Condition
        SAWcheck = SAWcond(k, rx, ry, rz, nrx, nry, nrz)
        if (SAWcheck == 0): continue
    
        #Move accepted if Both Conditions satisfied
        rx[k], ry[k], rz[k] = nrx, nry, nrz
        neaccept += 1

    #print('Percentage of accepted values = %.2f' % float(neaccept/eq_period*100.))

    '''Mean First-Passage Times of Looping of Polymers with Intrachain Reactive Monomers'''

    #revl[npaccept] = r + np.zeros((N, 3)) #Store 0th state in r-evolution

    while (trials < max_trials):
        trials += 1
        
        k = random.randint(0, Nm1) #Choose k-th atom for displacement
        dxyz, pm = random.choice(dirxyz), random.choice(dirud)
        nrx, nry, nrz = rx[k], ry[k], rz[k]
        if (dxyz == 0): nrx += pm * l
        elif (dxyz == 1): nry += pm * l
        else: nrz += pm * l
    
        #Bond-length condition along with no bond-crossing
        BLCcheck = BLC(k, rx, ry, rz, nrx, nry, nrz)
        if (BLCcheck == 0): continue
        
        #SAW Condition
        SAWcheck = SAWcond(k, rx, ry, rz, nrx, nry, nrz)
        if (SAWcheck == 0): continue
    
        #Move accepted if Both Conditions satisfied
        rx[k], ry[k], rz[k] = nrx, nry, nrz
        npaccept += 1
    
        #Check if Interacting Monomers came close
        ivec = [abs(rx[im1] - rx[im2]), abs(ry[im1] - ry[im2]), abs(rz[im1] - rz[im2])]
        ivec.sort()
        if (ivec == [0., 0., 2.*l]): break #If interacting monomers came close, stop
    
    if (trials == max_trials): print('Reached max. trials. Use higher time limits? Set higher max_trials!')
    
    nMCS = trials/N
    print('Number of MCS (Monte Carlo Steps) till interaction = ', nMCS)
    
    #Summing nMCS and storing in array in case of failure at some point
    TnMCS += nMCS
    arrnMCS.append(nMCS)

avgnMCS = TnMCS/max_iter

#Standard deviation error
sumerrsq = 0.
for i in range(len(arrnMCS)):
    sumerrsq += (arrnMCS[i] - avgnMCS)**2.
avgerrsq = sumerrsq/max_iter
err = np.sqrt(avgerrsq)

#Write datas to file
file1 = open("Datas/ freechain_%dmonomers_%d_looping_events.csv" % (N, max_iter), "w+")
for i in range(len(arrnMCS)):
    file1.write("%f\n" % (arrnMCS[i]))
file1.write("%f, %f" % (avgnMCS, err))
file1.close()

print('\nSimulation:\nFree-Chain Polymer with %d monomers.\nInteraction between %d-th and %d-th monomers.\n%d looping events.' % (N, im1 + 1, im2 + 1, max_iter))
print("\nAverage number of MCS (Monte Carlo Steps) till interaction = %f +/- %f" % (avgnMCS, err))
