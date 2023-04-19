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
  double xdiff, ydiff, zdiff, vmag, damp;
  // update position as per MD simulation
  //	apply constraints - push cloth outside of ball
  // swapped the looping order
  double c =  dt * 0.5 / mass;
  for (j = 0; n*n-4>=j; j+=4) {
      x[j] += dt * (vx[j] + c * fx[j]);
      x[j+1] += dt * (vx[j+1] + c*  fx[j+1]);
      x[j+2] += dt * (vx[j+2] + c * fx[j+2]);
      x[j+3] += dt * (vx[j+3] + c * fx[j+3]);
      oldfx[j] = fx[j];
      oldfx[j+1] = fx[j+1];
      oldfx[j+2] = fx[j+2];
      oldfx[j+3] = fx[j+3];
      y[j] += dt * (vy[j] + c * fy[j]);
      y[j+1] += dt * (vy[j+1] + c * fy[j+1]);
      y[j+2] += dt * (vy[j+2] + c * fy[j+2]);
      y[j+3] += dt * (vy[j+3] + c * fy[j+3]);
      oldfy[j] = fy[j];
      oldfy[j+1] = fy[j+1];
      oldfy[j+2] = fy[j+2];
      oldfy[j+3] = fy[j+3];
      z[j] += dt * (vz[j] + c * fz[j]);
      z[j+1] += dt * (vz[j+1] + c * fz[j+1]);
      z[j+2] += dt * (vz[j+2] + c * fz[j+2]);
      z[j+3] += dt * (vz[j+3] + c * fz[j+3]);
      oldfz[j] = fz[j];
      oldfz[j+1] = fz[j+1];
      oldfz[j+2] = fz[j+2];
      oldfz[j+3] = fz[j+3];
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
      xdiff = x[j+1] - xball;
      ydiff = y[j+1] - yball;
      zdiff = z[j+1] - zball;
      vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
      if (vmag < rball) {
        x[j+1] = xball + xdiff * rball / vmag;
        y[j+1] = yball + ydiff * rball / vmag;
        z[j+1] = zball + zdiff * rball / vmag;
        double proj_length = (vx[j+1]*xdiff + vy[j+1]*ydiff + vz[j+1]*zdiff)/vmag;
        vx[j+1]= (vx[j+1] - proj_length * xdiff/vmag)*0.1;
        vy[j+1]=(vy[j+1] - proj_length * ydiff/vmag)*0.1;
        vz[j+1]=(vz[j+1] - proj_length * zdiff/vmag)*0.1;
      }
      xdiff = x[j+2] - xball;
      ydiff = y[j+2] - yball;
      zdiff = z[j+2] - zball;
      vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
      if (vmag < rball) {
        x[j+2] = xball + xdiff * rball / vmag;
        y[j+2] = yball + ydiff * rball / vmag;
        z[j+2] = zball + zdiff * rball / vmag;
        double proj_length = (vx[j+2]*xdiff + vy[j+2]*ydiff + vz[j+2]*zdiff)/vmag;
        vx[j+2]= (vx[j+2] - proj_length * xdiff/vmag)*0.1;
        vy[j+2]=(vy[j+2] - proj_length * ydiff/vmag)*0.1;
        vz[j+2]=(vz[j+2] - proj_length * zdiff/vmag)*0.1;
      }
      xdiff = x[j+3] - xball;
      ydiff = y[j+3] - yball;
      zdiff = z[j+3] - zball;
      vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
      if (vmag < rball) {
        x[j+3] = xball + xdiff * rball / vmag;
        y[j+3] = yball + ydiff * rball / vmag;
        z[j+3] = zball + zdiff * rball / vmag;
        double proj_length = (vx[j+3]*xdiff + vy[j+3]*ydiff + vz[j+3]*zdiff)/vmag;
        vx[j+3]= (vx[j+3] - proj_length * xdiff/vmag)*0.1;
        vy[j+3]=(vy[j+3] - proj_length * ydiff/vmag)*0.1;
        vz[j+3]=(vz[j+3] - proj_length * zdiff/vmag)*0.1;
      }
    }
    while (j < n*n) {
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
      j+=1;
  }

  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  for (j = 0; n*n-4>=j; j+=4) {

    vx[j] = (vx[j] + c * (fx[j] + oldfx[j])) * damp;
    vx[j+1] = (vx[j+1] + c * (fx[j+1] + oldfx[j+1])) * damp;
    vx[j+2] = (vx[j+2] + c * (fx[j+2] + oldfx[j+2])) * damp;
    vx[j+3] = (vx[j+3] + c * (fx[j+3] + oldfx[j+3])) * damp;
    vy[j] = (vy[j] + c * (fy[j] + oldfy[j])) * damp;
    vy[j+1] = (vy[j+1] + c * (fy[j+1] + oldfy[j+1])) * damp;
    vy[j+2] = (vy[j+2] + c * (fy[j+2] + oldfy[j+2])) * damp;
    vy[j+3] = (vy[j+3] + c * (fy[j+3] + oldfy[j+3])) * damp;;
    vz[j] = (vz[j] + c * (fz[j] + oldfz[j])) * damp;
    vz[j+1] = (vz[j+1] + c * (fz[j+1] + oldfz[j+1])) * damp;
    vz[j+2] = (vz[j+2] + c * (fz[j+2] + oldfz[j+2])) * damp;
    vz[j+3] = (vz[j+3] + c * (fz[j+3] + oldfz[j+3])) * damp;
    *ke += vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];
    *ke += vx[j+1] * vx[j+1] + vy[j+1] * vy[j+1] + vz[j+1] * vz[j+1];
    *ke += vx[j+2] * vx[j+2] + vy[j+2] * vy[j+2] + vz[j+2] * vz[j+2];
    *ke += vx[j+3] * vx[j+3] + vy[j+3] * vy[j+3] + vz[j+3] * vz[j+3];

  }
  while (j<n*n){
    vx[j] = (vx[j] + c * (fx[j] + oldfx[j])) * damp;
    vy[j] = (vy[j] + c * (fy[j] + oldfy[j])) * damp;
    vz[j] = (vz[j] + c * (fz[j] + oldfz[j])) * damp;
    *ke += vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];
    j+=1;
  }
  *ke = *ke / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz) {
  double pe, rlen, xdiff, ydiff, zdiff, vmag, temp;
  // double t1,t2,t3;
  int nx, ny, dx, dy;

  pe = 0.0;
  // loop over particles
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      int nxny = nx * n + ny;
      fx[nxny] = 0.0;
      fy[nxny] = 0.0;
      fz[nxny] = -mass * grav;
      // loop over displacements
      for (dx = MAX(nx - delta, 0); dx < nx; dx++) {
        double nxdx = (nx - dx) * (nx - dx);
        for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4>=dy; dy+=4) {
        // for (dy = MAX(ny +delta-dx, 0); MIN(ny + delta + 1, n)-4>=dy; dy+=4) {
          // compute reference distance
          rlen = sqrt((double)(nxdx+ (ny - dy) * (ny - dy))) * sep;
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

          // compute reference distance
          rlen = sqrt((double)(nxdx + (ny - dy-1) * (ny - dy-1))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+1] - x[nxny];
          ydiff = y[dx * n + dy+1] - y[nxny];
          zdiff = z[dx * n + dy+1] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;
          // compute reference distance
          rlen = sqrt((double)(nxdx+ (ny - dy-2) * (ny - dy-2))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+2] - x[nxny];
          ydiff = y[dx * n + dy+2] - y[nxny];
          zdiff = z[dx * n + dy+2] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;

          // compute reference distance
          rlen = sqrt((double)(nxdx+ (ny - dy-3) * (ny - dy-3))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+3] - x[nxny];
          ydiff = y[dx * n + dy+3] - y[nxny];
          zdiff = z[dx * n + dy+3] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;
        }
        while (dy<MIN(ny + delta + 1, n)){
          rlen = sqrt((double)(nxdx+ (ny - dy) * (ny - dy))) * sep;
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
          dy+=1;
        }
      }
      for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
        // exclude self interaction
        if (nx != dx || ny != dy) {
          // compute reference distance
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
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
      dx+=1;
      for (; dx < MIN(nx + delta + 1, n); dx++) {
        for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4>=dy; dy+=4) {
          // compute reference distance
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
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

          // compute reference distance
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy-1) * (ny - dy-1))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+1] - x[nxny];
          ydiff = y[dx * n + dy+1] - y[nxny];
          zdiff = z[dx * n + dy+1] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;
          // compute reference distance
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy-2) * (ny - dy-2))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+2] - x[nxny];
          ydiff = y[dx * n + dy+2] - y[nxny];
          zdiff = z[dx * n + dy+2] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;

          // compute reference distance
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy-3) * (ny - dy-3))) * sep;
          // compute actual distance
          xdiff = x[dx * n + dy+3] - x[nxny];
          ydiff = y[dx * n + dy+3] - y[nxny];
          zdiff = z[dx * n + dy+3] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          temp = fcon * (vmag - rlen) / vmag;
          fx[nxny] += temp * xdiff;
          fy[nxny] += temp * ydiff;
          fz[nxny] += temp * zdiff;
        }
        while (dy<MIN(ny + delta + 1, n)){
          rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
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
          dy+=1;
        }
      }
    }
  }
  return 0.5 * pe;
}