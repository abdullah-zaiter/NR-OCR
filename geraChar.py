#% GERACHAR - ANN OCR 
#% 
#% ICIN/UnB Ago/2018 - Adolfo Bauchspiess
#% Generate Training Patterns 9x7


import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.lines as lines
from matplotlib.pyplot import *


def geraChar():
    P=np.zeros((63,16),dtype=int);
    T=np.eye(16,dtype=int);
    P[:,0]=[			#,A
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0
    ]
    P[:,1]=[			#,B
    1,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    1,1,1,1,1,1,0
    ]
    P[:,2]=[			#,C
    0,0,1,1,1,1,0,
    0,1,0,0,0,0,1,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,1,
    0,1,1,1,1,1,0
    ]
    P[:,3]=[			#,D
    1,1,1,1,1,1,0,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    0,1,0,0,0,0,1,
    1,1,1,1,1,1,0
    ]
    P[:,4]=[			#,E
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,0,
    1,0,0,0,1,0,0,
    1,1,1,1,1,0,0,
    1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,1,
    1,1,1,1,1,1,1
    ]
    P[:,5]=[			#,F
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,0,
    1,0,0,0,1,0,0,
    1,1,1,1,1,0,0,
    1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,1,0,0,0,0,0
    ]
    P[:,6]=[			#G
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,1,1,1,1,
    1,0,0,1,0,0,1,
    1,0,0,0,0,0,1,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0
    ]
    P[:,7]=[			#H
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1
    ]
    P[:,8]=[			#I
    0,1,1,1,1,1,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0
    ]
    P[:,9]=[			#J
    0,0,0,1,1,1,1,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    1,0,0,0,0,1,0,
    1,0,0,0,0,1,0,
    0,1,1,1,1,0,0
    ]
    P[:,10]=[			#K
    1,0,0,0,0,0,1,
    1,0,0,0,0,1,0,
    1,0,0,0,1,0,0,
    1,0,0,1,0,0,0,
    1,1,1,0,0,0,0,
    1,0,0,1,0,0,0,
    1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,
    1,0,0,0,0,0,1
    ]
    P[:,11]=[			#L
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,1,1,1,1,1,1    
    ]
    P[:,12]=[		#M
    1,0,0,0,0,0,1,
    1,1,0,0,0,1,1,
    1,0,1,0,1,0,1,
    1,0,0,1,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1
    ]
    P[:,13]=[		#N
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,0,0,0,0,1,
    1,0,1,0,0,0,1,
    1,0,0,1,0,0,1,
    1,0,0,0,1,0,1,
    1,0,0,0,0,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1    
    ]
    P[:,14]=[			#O
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0
    ]
    P[:,15]=[			#P
    1,1,1,1,1,1,0,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0
    ]
    P=P.transpose()   
    return (P,T)


def gchar_ruido(P, Ruido):
# Adds noide to the pattern P
# Returns the noise pattern Pn

    np.random.seed(int(time.clock()))

    Pn=np.zeros((63,16),dtype=int)
    
    # Copy function! P is only a pointer!!
    Pn=copiaV(P)
    
    BitsRuido=int(np.ceil(63*Ruido/100))
    noise_ind=np.zeros((BitsRuido,1),dtype=int)

    if BitsRuido == 0:
        return P
    
    for l in range(16):
        vals = np.random.rand(BitsRuido,1)*63
        for k in range(BitsRuido):
            noise_ind[k] = int(np.floor(vals[k]))
           
        for i in range(BitsRuido): 
            if Pn[l,noise_ind[i]] == 0: Pn[l,noise_ind[i]] = 1
            else: Pn[l,noise_ind[i]] = 0
        
    return Pn

def copia(x):
    y=x
    return y

def copiaV(P):
    f=np.vectorize(copia)
    return f(P)

def complement(x):
    return -x+1

def CompP(P):
    f=np.vectorize(complement)
    return f(P)

def showChar(P):
# Displays the 16 9x7 Characters

    plt.figure(figsize=(18,14))
    for i in range(16):
        plt.subplot(1,16,i+1)        
        Pc=CompP(P)
        plt.imshow(Pc[i,:].reshape(9,7),cmap=cm.Greys_r)
        plt.axis('off')    
    plt.show()
    return

def showchar(P):
    plt.rcParams['figure.facecolor'] = 'grey'
    plt.subplot(281); plt.imshow(np.reshape(255*P[0], (9, 7))); axis('off')
    plt.subplot(282); plt.imshow(np.reshape(255*P[1], (9, 7) )); axis('off')
    plt.subplot(283); plt.imshow(np.reshape(255*P[2], (9, 7) )); axis('off')
    plt.subplot(284); plt.imshow(np.reshape(255*P[3], (9, 7) )); axis('off')
    plt.subplot(285); plt.imshow(np.reshape(255*P[4], (9, 7) )); axis('off')
    plt.subplot(286); plt.imshow(np.reshape(255*P[5], (9, 7) )); axis('off')
    plt.subplot(287); plt.imshow(np.reshape(255*P[6], (9, 7) )); axis('off')
    plt.subplot(288); plt.imshow(np.reshape(255*P[7], (9, 7) )); axis('off')
    plt.subplot(289); plt.imshow(np.reshape(255*P[8], (9, 7) )); axis('off')
    plt.subplot(2,8,10); plt.imshow(np.reshape(255*P[9], (9, 7) )); axis('off')
    plt.subplot(2,8,11); plt.imshow(np.reshape(255*P[10], (9, 7) )); axis('off')
    plt.subplot(2,8,12); plt.imshow(np.reshape(255*P[11], (9, 7) )); axis('off')
    plt.subplot(2,8,13); plt.imshow(np.reshape(255*P[12], (9, 7) )); axis('off')
    plt.subplot(2,8,14); plt.imshow(np.reshape(255*P[13], (9, 7) )); axis('off')
    plt.subplot(2,8,15); plt.imshow(np.reshape(255*P[14], (9, 7) )); axis('off')
    plt.subplot(2,8,16); plt.imshow(np.reshape(255*P[15], (9, 7) )); axis('off')
    plt.show()

def validacao(Pn,net):
# calculate the output of the net for a given noise Patterns Pn
# bincon - incorrect Chars; idx - the recognized pattern (should be range(16))

    m=[]
    idx=[]

    # calculate predictions
    predictions = net.predict(Pn)
    for k in range(16):
       p = predictions[k,:]
       m.append(np.max(p))
       idx.append(np.argmax(p))
    
    # calculate number or incorrect characters
    bincor = 0; 
    for i in range(16): 
        if idx[i] <> i: bincor +=1;
            
    return (bincor,idx)   


