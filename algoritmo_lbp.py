import cv2 
import numpy as np
from collections import Counter 
import itertools 
import glob



# Metodo para scar el numero binario 
    
def numDecimal(binario):
    decimal = 0
    j = len(binario)-1
    for i in range(0,len(binario)):
        
        if(int(binario[i]) != 0):
            decimal = decimal + 2**j
        j = j-1
    return decimal 


def numBinario(P):
    numBinario = ""
    numBinario = numBinario + str(P[1][0]) + str(P[2][0]) + str(P[2][1]) + str(P[2][2]) +str(P[1][2]) + str(P[0][2]) + str(P[0][1]) + str(P[0][0])
    numBinario = numDecimal(numBinario)
    return numBinario


#Metodo para cambiar los vecinos segun el umbral 

def umbralMatriz(P,punto_central):
    Pr = P*0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if(P[i][j] >= punto_central):
                Pr[i][j] = 1
            else:
                Pr[i][j] = 0
            
    return Pr
            

# Método para calcular la vecindad 
def neighboor(p,P):
    N1=np.array([[0,0,0],[0,0,0],[0,0,0]])
    k=0
    l=0
    for i in range(p[0]-1,p[0]+2):
        for j in range(p[1]-1,p[1]+2):
            if i<0 or j<0 or i>len(P)-1 or j>len(P[0])-1:
                N1[k][l]=0
                l=l+1
            else:
                N1[k][l]=P[i][j]
                l=l+1
       
        k=k+1
        l=0
    N1 = umbralMatriz(N1,P[p[0]][p[1]])
    return N1

def miHPA(P):
    
    valor=0
    ck=[]
    totalP = len(P)*len(P[0])
    res = dict(Counter(sorted(itertools.chain(*P)))) 
    valores = list(res.values())
    llaves = list(res.keys())
    cont = 0
    for i in range(256):
        for j in range(len(llaves)):
            if(i == llaves[j]):
                cont = cont+valores[j]
                
                valor = cont
                
                valor = valor/totalP
                
        ck.append(valor)
        
    
    return ck    

def matrizBinariaDecimal(P):
    Pr = P*0
    for i in range(0,len(P)):
        for j in range(0,len(P[0])):
            matrizUm = (neighboor([i,j], P)) 
            Pr[i][j] = numBinario(matrizUm)
           
    return Pr

#Metodo para calcular los histogramas de la lista obtenida 
def calcularHistogramas(P):
    histogramas = []
   
    for i in range(len(P)):
        va = P[i]
        val = miHPA(va)
        histogramas.append(val)
    return histogramas

# Metodo para hacer un solo vector 
def unirHistogramas(P):
    #print(len(P))
    vectorUnion = []
    unirHistograma = []
    contador = 0
    fin = round(len(P)/64)
    for j in range(0,fin):
        
        for i in range(contador,contador+64):
            
            vectorUnion = vectorUnion+P[i]
        unirHistograma.append(vectorUnion)
        vectorUnion=[]
        contador= contador+64
    return unirHistograma
   
    
# Método para crear matriz de 16384 * 1 
def crearMatriz(histograma):
    P = np.zeros((len(histograma),len(histograma[0])))
    for j in range(len(histograma)):
        
        for i in range(len(histograma[0])):
            P[j][i] = histograma[j][i]
    return P
        


#path = glob.glob("imagenes/*.jpg")

path = glob.glob("ken.jpg")

cv_img = []
for img in path:
    n = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    n = cv2.resize(n, dsize=(240,240))
    cv_img.append(n)    

vectorBinario = []
for i in range(len(cv_img)):
    vectorBinario.append(matrizBinariaDecimal(cv_img[i]))
    
    
def histogramaImagenes(vectorBinario):
    
    ancho = len(vectorBinario[0])
    seAncho = round(ancho / 8)


    ## Generar las secuencias 
    secuenciaAncho = []
    secuenciaAlto = []

    for i in range(0,9):
        secuenciaAlto.append((seAncho*i))
        secuenciaAncho.append((seAncho*i)) 
    listaMatrices = []

    for k in range(len(vectorBinario)):
        for i in range(0,9):
            for j in range(0,9):
                if(i<8 and j<8):
                    print(str(secuenciaAlto[i])+"  "+str(secuenciaAlto[i+1])+"    "+str(secuenciaAlto[j])+" "+str(secuenciaAlto[j+1]))
                    listaMatrices.append(vectorBinario[k][secuenciaAlto[i]:secuenciaAlto[i+1],secuenciaAlto[j]:secuenciaAlto[j+1]])
    return listaMatrices

histogramaGImagenes = histogramaImagenes(vectorBinario)



histogramas = calcularHistogramas(histogramaGImagenes)
histogramaGeneral = unirHistogramas(histogramas)
matrizHisto = crearMatriz(histogramaGeneral)

    
   

