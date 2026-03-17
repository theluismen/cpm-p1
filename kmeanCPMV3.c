#include <stdlib.h>
#include <stdio.h>

#define N 600000
#define G 200

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
    int i, j, min, iter = 0;
    long dif, t, min_dif, curr_dif;
    long fS[G];

    do
    {
        for (i = 0; i < fK; i++)
        {
            fS[i] = 0;
            fA[i] = 0;
        }

        #pragma omp parallel for private(j, min, min_dif, curr_dif) reduction(+:fS[:fK], fA[:fK])
        for (i = 0; i < fN; i++)
        {
            min = 0;
            min_dif = labs(fV[i] - fR[0]);

            for (j = 1; j < fK; j++)
            {
                curr_dif = labs(fV[i] - fR[j]);
                if (curr_dif < min_dif)
                {
                    min = j;
                    min_dif = curr_dif;
                }
            }
            
            fS[min] += fV[i];
            fA[min] += 1;
        }

        dif = 0;

        #pragma omp parallel for reduction(+:dif) private(t)
        for (i = 0; i < fK; i++)
        {
            t = fR[i];

            if (fA[i])
            {
                fR[i] = fS[i] / fA[i];
            }
            dif += labs(t - fR[i]);
        }

        iter++;

    } while (dif);

    printf("iter %d\n", iter);
}

void qs(int ii, int fi, long fV[], int fA[])
{
    int i, f;
    long pi, pa, vtmp, vta, vfi, vfa;

    pi = fV[ii];
    pa = fA[ii];

    i = ii + 1;
    f = fi;

    vtmp = fV[i];
    vta = fA[i];

    while (i <= f)
    {
        if (vtmp < pi)
        {
            fV[i - 1] = vtmp;
            fA[i - 1] = vta;

            i++;

            vtmp = fV[i];
            vta = fA[i];
        }
        else
        {
            vfi = fV[f];
            vfa = fA[f];

            fV[f] = vtmp;
            fA[f] = vta;

            f--;

            vtmp = vfi;
            vta = vfa;
        }
    }

    fV[i - 1] = pi;
    fA[i - 1] = pa;

    if (ii < f)
        qs(ii, f, fV, fA);

    if (i < fi)
        qs(i, fi, fV, fA);
}

int main()
{
    int i;

    for (i = 0; i < N; i++)
        V[i] = (rand() % rand()) / N;

    // primeros candidatos
    for (i = 0; i < G; i++)
        R[i] = V[i];

    // calcular los G más representativos
    kmean(N, G, V, R, A);

    qs(0, G - 1, R, A);

    for (i = 0; i < G; i++)
        printf("R[%d] : %ld tiene %d agrupados\n", i, R[i], A[i]);

    return 0;
}