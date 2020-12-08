'''
Developer: Pallav Chanda <pallav16@iiserb.ac.in>; 16131, BSMS 2016; IISER Bhopal
Month: December 2019 (Start Date: 07.12.19)
Code written during a short-term project under Prof. Snigdha Thakur <snigdha@iiserb.ac.in>, IISER Bhopal
Code based on:

'''

import numpy as np
import matplotlib.pyplot as plt

####---------------AIM OF THE CODE---------------####

print('\n\nThis code simulates looping events for Free-Chain Polymers.\nReturns the average number of MCS (Monte Carlo Steps) the algorithm takes till interaction.\n')

####------------------VARIABLES------------------####

l = 1.
N = int(input('Enter number of monomers in the polymer: ')) #chain length (number of monmers) of the polymer
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
max_trials = 10000000 #Prevent infinite looping

#Positions of interacting monomers
im1, im2 = int(input('\nEnter position of 1st interacting monomer: ')), int(input('Enter position of 2nd interacting monomer: '))
print('\nYou want: Interaction between %d-th and %d-th monomers.\n' % (im1, im2))

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
Bset = np.array([[0., 0., 2.], [0., 1., 2.], [1., 1., 2.], [1., 2., 2.], [0., 0., 3.], [0., 1., 3.]])

#Direction arrays for choosing random direction to move
dirxyz = [0, 1, 2] #direction to move in. 0=x, 1=y, 2=z.
dirud = [1, -1] #direction to move after choozing x/y/z. +1=+ve, -1=-ve

#Write datas to file
file1 = open("freechain_%dmonomers_%d_looping events" % (N, max_iter), "w+")

####-------------FUNCTION DEFINITIONS------------####

#Function to check SAW condition (Each lattice site can be occupied by only one monomer)
def SAWcond(k_, newr_):
    satisfied = 0
    for j in range(N):
        if (j == k_): continue
        dist = abs(np.subtract(newr_[j], newr_[k_]))
        for i in range(3):
            if (dist[i] >= 2.*l):
                satisfied += 1
                break
    if (satisfied == N-1): return 1
    else: return 0

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
    r = np.zeros((N, 3)) #The polymer #r[monomer][coordinates]

    #Generating initial modeled chain
    dxyz = np.random.choice(dirxyz)
    r[1][dxyz] = np.random.choice(dirud) * 2.*l
    i = 2
    while i < N:
        r[i] = r[i-1]
        dxyz = np.random.choice(dirxyz)
        r[i][dxyz] = r[i][dxyz] + (np.random.choice(dirud) * 2.*l)
        for j in range(i):
            if np.all(r[j] == r[i]):
                i -= 1
                continue
        i += 1

    #Initializing arrays
    bvec1, bvec2 = np.zeros(3), np.zeros(3) #Bond vectors
    newr = np.zeros((N, 3)) #Temporary storage of r for checking conditions
    #revl = np.zeros((int(max_trials/3), N, 3)) #r-evolution. Stores all r after equilibration #revl[accepted state][monomer][coordinates]
    ivec = np.zeros(3) #Interaction vector to check if the interacting monomers came close
    
    '''####----Algorithm implementation starts here----####'''
    
    '''Monte Carlo Equilibration of Modeled Chain'''

    for i in range(eq_period):
        newr = r + np.zeros((N, 3))
        k = np.random.randint(0, N) #Choose k-th atom for displacement
        dxyz = np.random.choice(dirxyz)
        newr[k][dxyz] = newr[k][dxyz] + (np.random.choice(dirud) * l)
        
        #SAW Condition
        SAWcheck = SAWcond(k, newr)
        if (SAWcheck == 0): continue
        
        #Bond-length condition along with no bond-crossing
        inset = 0
        if (k == 0):
            bvec2 = np.sort(abs(np.subtract(newr[k], newr[k+1])))
            if (all(bvec2 in sublist for sublist in Bset)): inset = 1
        elif (k == N-1):
            bvec1 = np.sort(abs(np.subtract(newr[k], newr[k-1])))
            if (all(bvec1 in sublist for sublist in Bset)): inset = 1
        else:
            bvec1 = np.sort(abs(np.subtract(newr[k], newr[k-1])))
            bvec2 = np.sort(abs(np.subtract(newr[k], newr[k+1])))
            if (all(bvec1 in sublist for sublist in Bset) and all(bvec2 in sublist for sublist in Bset)): inset = 1
        if (inset == 0): continue
        
        #Move accepted if Both Conditions satisfied
        r = newr + np.zeros((N, 3))
        neaccept += 1
    
    #print('Percentage of accepted values = %.2f' % float(neaccept/eq_period*100.))

    '''Mean First-Passage Times of Looping of Polymers with Intrachain Reactive Monomers'''

    #revl[npaccept] = r + np.zeros((N, 3)) #Store 0th state in r-evolution

    while (trials < max_trials):
        trials += 1
        newr = r + np.zeros((N, 3))
        k = np.random.randint(0, N) #Choose k-th atom for displacement
        dxyz = np.random.choice(dirxyz)
        newr[k][dxyz] = newr[k][dxyz] + (np.random.choice(dirud) * l)
        
        #SAW Condition
        SAWcheck = SAWcond(k, newr)
        if (SAWcheck == 0): continue #reject if not satisfied
        
        #Bond-length condition along with no bond-crossing
        inset = 0
        if (k == 0):
            bvec2 = np.sort(abs(np.subtract(newr[k], newr[k+1])))
            if (all(bvec2 in sublist for sublist in Bset)): inset = 1
        elif (k == N-1):
            bvec1 = np.sort(abs(np.subtract(newr[k], newr[k-1])))
            if (all(bvec1 in sublist for sublist in Bset)): inset = 1
        else:
            bvec1 = np.sort(abs(np.subtract(newr[k], newr[k-1])))
            bvec2 = np.sort(abs(np.subtract(newr[k], newr[k+1])))
            if (all(bvec1 in sublist for sublist in Bset) and all(bvec2 in sublist for sublist in Bset)): inset = 1
        if (inset == 0): continue #reject if not satisfied
        
        #Move accepted if Both Conditions are satisfied
        r = newr + np.zeros((N, 3))
        npaccept += 1
        #revl[npaccept] = r + np.zeros((N, 3)) #Store in r-evolution
        
        #Check if Interacting Monomers came close
        ivec = np.subtract(r[im1], r[im2])
        if (np.linalg.norm(ivec) == blmin): break #If interacting monomers came close, stop
    
    if (trials == max_trials): print('Reached max. trials. Use higher time limits? Set higher max_trials')
    nMCS = trials/N
    print('Number of MCS (Monte Carlo Steps) till interaction = ', nMCS)
    #print('Max state = ', npaccept)
    
    #Summing nMCS and storing in array in case of failure at some point
    TnMCS += nMCS
    arrnMCS.append(nMCS)

file1.writelines(arrnMCS)
file1.close()

avgnMCS = TnMCS/max_iter
print('\nSimulation:\nFree-Chain Polymer with %d monomers.\nInteraction between %d-th and %d-th monomers.\n%d looping events.' % (N, im1, im2, max_iter))
print('\nAverage number of MCS (Monte Carlo Steps) till interaction = ', avgnMCS)
