#include <stdlib.h>
#include <stdio.h>
#include <omp.h> // Incloure la llibreria OpenMP

#define N 600000
#define G 200  

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
    int iter = 0;
    long dif;
    static int fD[N]; // Estático para no sobrecargar la pila

    do {
        long fS_global[G] = {0};
        int fA_global[G] = {0};
        dif = 0;

        #pragma omp parallel
        {
            // Arrays locales inicializados a cero [cite: 194, 218]
            long l_S[G];
            int l_A[G];
            for(int i = 0; i < fK; i++) { l_S[i] = 0; l_A[i] = 0; }

            #pragma omp for schedule(static)
            for (int i = 0; i < fN; i++) {
                const long val = fV[i]; // Registro local [cite: 179]
                int best_j = 0;
                
                // Calculamos el primero fuera para inicializar min_d
                long min_d = val - fR[0];
                if (min_d < 0) min_d = -min_d;

                // Bucle optimizado para vectorización
                for (int j = 1; j < fK; j++) {
                    long cur_d = val - fR[j];
                    if (cur_d < 0) cur_d = -cur_d;
                    
                    if (cur_d < min_d) {
                        min_d = cur_d;
                        best_j = j;
                    }
                }
                
                fD[i] = best_j;
                l_S[best_j] += val;
                l_A[best_j]++;
            }

            // Consolidación final por hilo [cite: 147]
            #pragma omp critical
            {
                for (int j = 0; j < fK; j++) {
                    fS_global[j] += l_S[j];
                    fA_global[j] += l_A[j];
                }
            }
        } // Fin de parallel [cite: 42, 43]

        // Actualización de centroides y cálculo de dif
        for(int j = 0; j < fK; j++) {
            if (fA_global[j] > 0) {
                long antiguo = fR[j];
                fR[j] = fS_global[j] / fA_global[j];
                fA[j] = fA_global[j]; // Guardamos el conteo final
                
                long d = antiguo - fR[j];
                if (d < 0) d = -d;
                dif += d;
            }
        }
        iter++;
    } while(dif > 0);

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