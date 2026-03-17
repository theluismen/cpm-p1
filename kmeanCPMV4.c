#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 600000
#define G 200  

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
    int i, j, iter = 0;
    long dif, t;
    long fS[G];
    int fD[N]; 

    do {
        for(i = 0; i < fK; i++) {
            fS[i] = 0;
            fA[i] = 0;
        }

        #pragma omp parallel for schedule(static) private(j) reduction(+:fS[0:fK], fA[0:fK])
        for (i = 0; i < fN; i++) {
            int min = 0;
            long dif_local = labs(fV[i] - fR[0]); 
            for (j = 1; j < fK; j++) {
                long curr_dif = labs(fV[i] - fR[j]);
                if (curr_dif < dif_local) {
                    min = j;
                    dif_local = curr_dif;
                }
            }
            fD[i] = min;
            fS[min] += fV[i];
            fA[min]++;
        }

        dif = 0;
        #pragma omp parallel for reduction(+:dif) private(t)
        for(i = 0; i < fK; i++) {
            t = fR[i];
            if (fA[i]) fR[i] = fS[i] / fA[i];
            dif += labs(t - fR[i]);
        }
        
        iter++;
    } while(dif);

    printf("iter %d\n", iter);
}

// ... La funció qs() (QuickSort) i main() es mantenen exactament igual ...
void qs(int ii, int fi, long fV[], int fA[]) {
    // [Codi original mantingut]
    int i,f,j;
    long pi,pa,vtmp,vta,vfi,vfa;

    pi = fV[ii]; pa = fA[ii];
    i = ii +1; f = fi;
    vtmp = fV[i]; vta = fA[i];

    while (i <= f) {
        if (vtmp < pi) {
            fV[i-1] = vtmp; fA[i-1] = vta; i ++;
            vtmp = fV[i]; vta = fA[i];
        } else {
            vfi = fV[f]; vfa = fA[f];
            fV[f] = vtmp; fA[f] = vta; f --;
            vtmp = vfi; vta = vfa;
        }
    }
    fV[i-1] = pi; fA[i-1] = pa;

    if (ii < f) qs(ii,f,fV,fA);
    if (i < fi) qs(i,fi,fV,fA);
}

int main() {
    int i;
    // Generació aleatòria
    for (i=0;i<N;i++) V[i] = (rand()%rand())/N;
    // primers candidats
    for (i=0;i<G;i++) R[i] = V[i];
    // calcular els G mes representatius
    kmean(N,G,V,R,A);
    qs(0,G-1,R,A);

    for (i=0;i<G;i++) 
        printf("R[%d] : %ld te %d agrupats\n",i,R[i],A[i]);

    return(0);
}