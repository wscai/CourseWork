#include "./cloth_code.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int UNUSED(delta), double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz) {
  int i, nx, ny;

  // Free any existing
  free(*x);
  free(*y);
  free(*z);
  free(*cpx);
  free(*cpy);
  free(*cpz);

  // allocate arrays to hold locations of nodes
  *x = (double *)malloc(n * n * sizeof(double));
  *y = (double *)malloc(n * n * sizeof(double));
  *z = (double *)malloc(n * n * sizeof(double));
  // This is for opengl stuff
  *cpx = (double *)malloc(n * n * sizeof(double));
  *cpy = (double *)malloc(n * n * sizeof(double));
  *cpz = (double *)malloc(n * n * sizeof(double));

  // initialize coordinates of cloth
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      (*x)[n * nx + ny] = nx * sep - (n - 1) * sep * 0.5 + offset;
      (*z)[n * nx + ny] = rball + 1;
      (*y)[n * nx + ny] = ny * sep - (n - 1) * sep * 0.5 + offset;
      (*cpx)[n * nx + ny] = 0;
      (*cpz)[n * nx + ny] = 1;
      (*cpy)[n * nx + ny] = 0;
    }
  }

  // Throw away existing arrays
  free(*fx);
  free(*fy);
  free(*fz);
  free(*vx);
  free(*vy);
  free(*vz);
  free(*oldfx);
  free(*oldfy);
  free(*oldfz);
  // Alloc new
  *fx = (double *)malloc(n * n * sizeof(double));
  *fy = (double *)malloc(n * n * sizeof(double));
  *fz = (double *)malloc(n * n * sizeof(double));
  *vx = (double *)malloc(n * n * sizeof(double));
  *vy = (double *)malloc(n * n * sizeof(double));
  *vz = (double *)malloc(n * n * sizeof(double));
  *oldfx = (double *)malloc(n * n * sizeof(double));
  *oldfy = (double *)malloc(n * n * sizeof(double));
  *oldfz = (double *)malloc(n * n * sizeof(double));
  for (i = 0; i < n * n; i++) {
    (*vx)[i] = 0.0;
    (*vy)[i] = 0.0;
    (*vz)[i] = 0.0;
    (*fx)[i] = 0.0;
    (*fy)[i] = 0.0;
    (*fz)[i] = 0.0;
  }
}

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball,
              double zball, double dt, double *x, double *y, double *z,
              double *fx, double *fy, double *fz, double *vx, double *vy,
              double *vz, double *oldfx, double *oldfy, double *oldfz,
              double *pe, double *ke, double *te) {
  int j;
  double damp = 0.995;
  // update position as per MD simulation
  //	apply constraints - push cloth outside of ball
  // swapped the looping order
  double c =  dt * 0.5 / mass;
  double vmag,xdiff,ydiff,zdiff;
  // double vmag[n*n] = {0};
  // double xdiff[n*n] = {0};
  // double ydiff[n*n] = {0};
  // double zdiff[n*n] = {0};
  #pragma omp simd private(vmag,xdiff,ydiff,zdiff)
  for (j = 0; j<n*n; j++) {
    x[j] += dt * (vx[j] + c * fx[j]);
    oldfx[j] = fx[j];
    y[j] += dt * (vy[j] + c * fy[j]);
    oldfy[j] = fy[j];
    z[j] += dt * (vz[j] + c * fz[j]);
    oldfz[j] = fz[j];
    xdiff = x[j] - xball;
    ydiff = y[j] - yball;
    zdiff = z[j] - zball;
    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
    if (vmag < rball) {
      x[j] = xball + xdiff * rball / vmag;
      y[j] = yball + ydiff * rball / vmag;
      z[j] = zball + zdiff * rball / vmag;
      double proj_length = (vx[j]*xdiff + vy[j]*ydiff + vz[j]*zdiff)/vmag;
      vx[j]= (vx[j] - proj_length * xdiff/vmag)*0.1;
      vy[j]=(vy[j] - proj_length * ydiff/vmag)*0.1;
      vz[j]=(vz[j] - proj_length * zdiff/vmag)*0.1;
    }
  }

  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

  // Add a damping factor to eventually set velocity to zero

  *ke = 0.0;
  double k = 0;
  #pragma omp simd reduction(+:k)
  for (j = 0; j<n*n; j++) {
    vx[j] = (vx[j] + c * (fx[j] + oldfx[j])) * damp;
    vy[j] = (vy[j] + c * (fy[j] + oldfy[j])) * damp;
    vz[j] = (vz[j] + c * (fz[j] + oldfz[j])) * damp;
    k += vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];
  }
  *ke = k / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz) {
  double pe, rlen, xdiff, ydiff, zdiff, vmag, temp;
  int nx, ny, dx, dy;

  pe = 0.0;
  // loop over particles
  for (ny = 0; ny < n; ny++) {
    for (nx = 0; nx < n; nx++) {
      int nxny = nx * n + ny;
      fx[nxny] = 0.0;
      fy[nxny] = 0.0;
      fz[nxny] = -mass * grav;

      // loop over displacements
      for (dx = MAX(nx - delta, 0); dx < nx; dx++) {
        double nxdx = (nx - dx) * (nx - dx);
        #pragma omp simd reduction(+:fx[nxny],fy[nxny],fz[nxny],pe)
        for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
          // exclude self interaction
            // compute reference distance
            rlen = sqrt(double(nxdx + (ny - dy) * (ny - dy))) * sep;
            // compute actual distance
            xdiff = x[dx * n + dy] - x[nxny];
            ydiff = y[dx * n + dy] - y[nxny];
            zdiff = z[dx * n + dy] - z[nxny];
            vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            // potential energy and force
            pe += fcon * (vmag - rlen) * (vmag - rlen);
            temp = fcon * (vmag - rlen) / vmag;
            fx[nxny] += temp * xdiff;
            fy[nxny] += temp * ydiff;
            fz[nxny] += temp * zdiff;
        }
      }
      double nxdx = (nx - dx) * (nx - dx);
      #pragma omp simd reduction(+:fx[nxny],fy[nxny],fz[nxny],pe)
        for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
          // exclude self interaction
          if (nx != dx || ny != dy) {
            // compute reference distance
            rlen =
                sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) *
                sep;
            // compute actual distance
            xdiff = x[dx * n + dy] - x[nx * n + ny];
            ydiff = y[dx * n + dy] - y[nx * n + ny];
            zdiff = z[dx * n + dy] - z[nx * n + ny];
            vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            // potential energy and force
            pe += fcon * (vmag - rlen) * (vmag - rlen);
            temp = fcon * (vmag - rlen) / vmag;
            fx[nxny] += temp * xdiff;
            fy[nxny] += temp * ydiff;
            fz[nxny] += temp * zdiff;
          }
        }
      dx++;
      for (; dx < MIN(nx + delta + 1, n); dx++) {
        double nxdx = (nx - dx) * (nx - dx);
        #pragma omp simd reduction(+:fx[nxny],fy[nxny],fz[nxny],pe)
        for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
          // exclude self interaction
            // compute reference distance
            rlen = sqrt(double(nxdx + (ny - dy) * (ny - dy))) * sep;
            // compute actual distance
            xdiff = x[dx * n + dy] - x[nxny];
            ydiff = y[dx * n + dy] - y[nxny];
            zdiff = z[dx * n + dy] - z[nxny];
            vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            // potential energy and force
            pe += fcon * (vmag - rlen) * (vmag - rlen);
            temp = fcon * (vmag - rlen) / vmag;
            fx[nxny] += temp * xdiff;
            fy[nxny] += temp * ydiff;
            fz[nxny] += temp * zdiff;
        }
      }
    }
  }
  return 0.5 * pe;
}
