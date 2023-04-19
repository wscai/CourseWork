#include "./cloth_code.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

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
  __m256d cc = _mm256_set1_pd(c);
  __m256d dtt = _mm256_set1_pd(dt);
  #pragma omp parallel for private(xdiff,ydiff,zdiff,vmag) schedule(static,n)
  for (j = 0; n*n-4>=j; j+=4) {
    __m256d vvv = _mm256_loadu_pd(&(vx[j]));
    __m256d fff = _mm256_loadu_pd(&(fx[j]));
    __m256d xxx = _mm256_loadu_pd(&(x[j]));
    vvv = _mm256_mul_pd(dtt,_mm256_add_pd(vvv,_mm256_mul_pd(cc,fff)));
    xxx = _mm256_add_pd(xxx,vvv);
    _mm256_storeu_pd(&x[j],xxx);
    _mm256_storeu_pd(&oldfx[j],fff);
    __m256d xdifff = _mm256_sub_pd(xxx,_mm256_set1_pd(xball));
    xxx = _mm256_loadu_pd(&(y[j]));
    fff = _mm256_loadu_pd(&(fy[j]));
    vvv = _mm256_loadu_pd(&(vy[j]));
    vvv = _mm256_mul_pd(dtt,_mm256_add_pd(vvv,_mm256_mul_pd(cc,fff)));
    xxx = _mm256_add_pd(xxx,vvv);
    _mm256_storeu_pd(&y[j],xxx);
    _mm256_storeu_pd(&oldfy[j],fff);
    __m256d ydifff = _mm256_sub_pd(xxx,_mm256_set1_pd(yball));
    xxx = _mm256_loadu_pd(&(z[j]));
    fff = _mm256_loadu_pd(&(fz[j]));
    vvv = _mm256_loadu_pd(&(vz[j]));
    vvv = _mm256_mul_pd(dtt,_mm256_add_pd(vvv,_mm256_mul_pd(cc,fff)));
    xxx = _mm256_add_pd(xxx,vvv);
    _mm256_storeu_pd(&z[j],xxx);
    _mm256_storeu_pd(&oldfz[j],fff);
    __m256d zdifff = _mm256_sub_pd(xxx,_mm256_set1_pd(zball));
    __m256d vmagg = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(xdifff,xdifff),_mm256_add_pd(_mm256_mul_pd(ydifff,ydifff),_mm256_mul_pd(zdifff,zdifff))));
    if (vmagg[0] < rball) {
      x[j] = xball + xdifff[0] * rball / vmagg[0];
      y[j] = yball + ydifff[0] * rball / vmagg[0];
      z[j] = zball + zdifff[0] * rball / vmagg[0];
      double proj_length = (vx[j]*xdifff[0] + vy[j]*ydifff[0] + vz[j]*zdifff[0])/vmagg[0];
      vx[j]= (vx[j] - proj_length * xdifff[0]/vmagg[0])*0.1;
      vy[j]=(vy[j] - proj_length * ydifff[0]/vmagg[0])*0.1;
      vz[j]=(vz[j] - proj_length * zdifff[0]/vmagg[0])*0.1;
    }
    if (vmagg[1] < rball) {
      x[j+1] = xball + xdifff[1] * rball / vmagg[1];
      y[j+1] = yball + ydifff[1] * rball / vmagg[1];
      z[j+1] = zball + zdifff[1] * rball / vmagg[1];
      double proj_length = (vx[j+1]*xdifff[1] + vy[j+1]*ydifff[1] + vz[j+1]*zdifff[1])/vmagg[1];
      vx[j+1]= (vx[j+1] - proj_length * xdifff[1]/vmagg[1])*0.1;
      vy[j+1]=(vy[j+1] - proj_length * ydifff[1]/vmagg[1])*0.1;
      vz[j+1]=(vz[j+1] - proj_length * zdifff[1]/vmagg[1])*0.1;
    }
    if (vmagg[2] < rball) {
      x[j+2] = xball + xdifff[2] * rball / vmagg[2];
      y[j+2] = yball + ydifff[2] * rball / vmagg[2];
      z[j+2] = zball + zdifff[2] * rball / vmagg[2];
      double proj_length = (vx[j+2]*xdifff[2] + vy[j+2]*ydifff[2] + vz[j+2]*zdifff[2])/vmagg[2];
      vx[j+2]= (vx[j+2] - proj_length * xdifff[2]/vmagg[2])*0.1;
      vy[j+2]=(vy[j+2] - proj_length * ydifff[2]/vmagg[2])*0.1;
      vz[j+2]=(vz[j+2] - proj_length * zdifff[2]/vmagg[2])*0.1;
    }
    if (vmagg[3] < rball) {
      x[j+3] = xball + xdifff[3] * rball / vmagg[3];
      y[j+3] = yball + ydifff[3] * rball / vmagg[3];
      z[j+3] = zball + zdifff[3] * rball / vmagg[3];
      double proj_length = (vx[j+3]*xdifff[3] + vy[j+3]*ydifff[3] + vz[j+3]*zdifff[3])/vmagg[3];
      vx[j+3]= (vx[j+3] - proj_length * xdifff[3]/vmagg[3])*0.1;
      vy[j+3]=(vy[j+3] - proj_length * ydifff[3]/vmagg[3])*0.1;
      vz[j+3]=(vz[j+3] - proj_length * zdifff[3]/vmagg[3])*0.1;
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
  double k = 0;
  __m256d dampp = _mm256_set1_pd(damp);
  #pragma omp parallel for reduction(+:k)
  for (j = 0; n*n-4>=j; j+=4) {
    double kke[4] = {0};
    __m256d vvv = _mm256_loadu_pd(&(vx[j]));
    __m256d ooo = _mm256_loadu_pd(&oldfx[j]);
    __m256d fff = _mm256_loadu_pd(&fx[j]);
    vvv = _mm256_mul_pd(_mm256_add_pd(vvv, _mm256_mul_pd(_mm256_add_pd(fff,ooo),cc)), dampp);
    _mm256_storeu_pd(&vx[j],vvv);
    __m256d vvvy = _mm256_loadu_pd(&(vy[j]));
    ooo = _mm256_loadu_pd(&oldfy[j]);
    fff = _mm256_loadu_pd(&fy[j]);
    vvvy = _mm256_mul_pd(_mm256_add_pd(vvvy, _mm256_mul_pd(_mm256_add_pd(fff,ooo),cc)), dampp);
    _mm256_storeu_pd(&vy[j],vvvy);
    __m256d vvvz = _mm256_loadu_pd(&(vz[j]));
    ooo = _mm256_loadu_pd(&oldfz[j]);
    fff = _mm256_loadu_pd(&fz[j]);
    vvvz = _mm256_mul_pd(_mm256_add_pd(vvvz, _mm256_mul_pd(_mm256_add_pd(fff,ooo),cc)), dampp);
    _mm256_storeu_pd(&vz[j],vvvz);
    vvvz = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vvv,vvv), _mm256_mul_pd(vvvy,vvvy)), _mm256_mul_pd(vvvz,vvvz));
    vvvz = _mm256_hadd_pd(vvvz,vvvz);
    k += vvvz[0]+vvvz[2];
  }
  *ke=k;
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
  // int nx, ny;
  int dx, dy;
  int nxny;
  
  pe = 0.0;
  double pee[4] = {0};
  // loop over particles
  __m256d sepp = _mm256_set1_pd(sep);
  __m256d fconn = _mm256_set1_pd(fcon);
  // #pragma omp parallel for reduction(+:pe) collapse(2)
  // for (nx = 0; nx < n; nx++) {
  //   for (ny = 0; ny < n; ny++) {
  // #pragma omp parallel for reduction(+:pe) schedule(static,n)

  #pragma omp parallel for reduction(+:pe) private(rlen, xdiff, ydiff, zdiff, vmag, temp) schedule(static,n)
  for (nxny = 0; nxny<n*n; nxny++){
      // int nxny = nx * n + ny;
      int nx = nxny/n;
      int ny = nxny- nx*n;

      fx[nxny] = 0.0;
      fy[nxny] = 0.0;
      fz[nxny] = -mass * grav;
      __m256d xxxx = _mm256_set1_pd(x[nxny]);
      __m256d yyyy = _mm256_set1_pd(y[nxny]);
      __m256d zzzz = _mm256_set1_pd(z[nxny]);

      for (dx = MAX(nx - delta, 0); dx < nx; dx++) {
        double idxx = double((nx - dx) * (nx - dx));
        __m256d nxdx = _mm256_set1_pd(idxx); 

        for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4>=dy; dy+=4) {
          // compute reference distance
          __m256d nydy = _mm256_set_pd((ny - dy-3.0) * (ny - dy -3.0),(ny - dy-2.0) * (ny - dy-2.0), (ny - dy-1.0) * (ny - dy -1.0), double((ny - dy) * (ny - dy)));
          __m256d rlenn = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_add_pd(nxdx,nydy)),sepp);
          __m256d xdifff = _mm256_sub_pd(_mm256_loadu_pd(&(x[dx * n + dy])),xxxx);
          __m256d ydifff = _mm256_sub_pd(_mm256_loadu_pd(&(y[dx * n + dy])),yyyy);
          __m256d zdifff = _mm256_sub_pd(_mm256_loadu_pd(&(z[dx * n + dy])),zzzz);
          __m256d vmagg = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(xdifff,xdifff),_mm256_mul_pd(ydifff,ydifff)), _mm256_mul_pd(zdifff,zdifff)));
          __m256d vmaggrlenn = _mm256_sub_pd(vmagg,rlenn);
          __m256d peee = _mm256_mul_pd(fconn,_mm256_mul_pd(vmaggrlenn,vmaggrlenn));
          __m256d temp = _mm256_mul_pd(fconn,_mm256_div_pd(vmaggrlenn,vmagg));
          __m256d ans = _mm256_hadd_pd(peee,_mm256_mul_pd(temp,xdifff));
          pe+=ans[0]+ans[2];
          fx[nxny]+=ans[1]+ans[3];
          ans = _mm256_hadd_pd(_mm256_mul_pd(temp,ydifff),_mm256_mul_pd(temp,zdifff));
          fy[nxny]+=ans[0]+ans[2];
          fz[nxny]+=ans[1]+ans[3];
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
      for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
        // exclude self interaction
        if (nx != dx || ny != dy) {
          // compute reference distance
          rlen = sqrt((double)((ny - dy) * (ny - dy))) * sep;
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
        double idxx = double((nx - dx) * (nx - dx));
        __m256d nxdx = _mm256_set1_pd(idxx);
        for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4>=dy; dy+=4) {
          // compute reference distance
          __m256d nydy = _mm256_set_pd((ny - dy-3.0) * (ny - dy -3.0),(ny - dy-2.0) * (ny - dy-2.0), (ny - dy-1.0) * (ny - dy -1.0), double((ny - dy) * (ny - dy)));
          __m256d rlenn = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_add_pd(nxdx,nydy)),sepp);
          __m256d xdifff = _mm256_sub_pd(_mm256_loadu_pd(&(x[dx * n + dy])),xxxx);
          __m256d ydifff = _mm256_sub_pd(_mm256_loadu_pd(&(y[dx * n + dy])),yyyy);
          __m256d zdifff = _mm256_sub_pd(_mm256_loadu_pd(&(z[dx * n + dy])),zzzz);
          __m256d vmagg = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(xdifff,xdifff),_mm256_mul_pd(ydifff,ydifff)), _mm256_mul_pd(zdifff,zdifff)));
          __m256d vmaggrlenn = _mm256_sub_pd(vmagg,rlenn);
          __m256d peee = _mm256_mul_pd(fconn,_mm256_mul_pd(vmaggrlenn,vmaggrlenn));
          __m256d temp = _mm256_mul_pd(fconn,_mm256_div_pd(vmaggrlenn,vmagg));
          __m256d ans = _mm256_hadd_pd(peee,_mm256_mul_pd(temp,xdifff));
          pe+=ans[0]+ans[2];
          fx[nxny]+=ans[1]+ans[3];
          ans = _mm256_hadd_pd(_mm256_mul_pd(temp,ydifff),_mm256_mul_pd(temp,zdifff));
          fy[nxny]+=ans[0]+ans[2];
          fz[nxny]+=ans[1]+ans[3];
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
  // }
  return 0.5 * pe;
}
