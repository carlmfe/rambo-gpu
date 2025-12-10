#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <stdlib.h>

#include "rng.h"
#include "rambo.h"

using namespace std;

void rambo(double et, vector<double>& xm, XorShift64State& rng, vector<double*> momenta_out, double& wt) {
/**********************************************************************
 *                       rambo                                         *
 *    ra(ndom)  m(omenta)  b(eautifully)  o(rganized)                  *
 *                                                                     *
 *    a democratic multi-particle phase space generator                *
 *    authors:  s.d. ellis,  r. kleiss,  w.j. stirling                 *
 *    this is version 1.0 -  written by r. kleiss                      *
 *    -- adjusted by hans kuijf, weights are logarithmic (20-08-90)    *
 *                                                                     *
 *    n  = number of particles                                         *
 *    et = total centre-of-mass energy                                 *
 *    xm = particle masses ( dim=nexternal-nincoming )                 *
 *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
 *    wt = weight of the event                                         *
 ***********************************************************************/
  int n = xm.size();
  vector<double*> q, p;
  vector<double> z(n), r(4), b(3), p2(n), xm2(n), e(n), v(n);
  static vector<int> iwarn(5, 0);
  static double acc = 1e-14;
  static int itmax = 6,ibegin = 0;
  static double twopi=8.*atan(1.);
  static double po2log=log(twopi/4.);

  for(int i=0; i < n; i++){
    q.push_back(new double[4]);
    p.push_back(new double[4]);
  }
// initialization step: factorials for the phase space weight
  if(ibegin==0){
    ibegin=1;
    z[1]=po2log;
    for(int k=2;k < n;k++)
      z[k]=z[k-1]+po2log-2.*log(double(k-1));
    for(int k=2;k < n;k++)
      z[k]=(z[k]-log(double(k)));
  }
// check on the number of particles
  if(n<1 || n>101){
    cout << "Too few or many particles: " << n << endl;
    return;
  }
// check whether total energy is sufficient; count nonzero masses
  double xmt=0.;
  int nm=0;
  for(int i=0; i<n; i++){
    if(xm[i]!=0.) nm=nm+1;
    xmt=xmt+abs(xm[i]);
  }
  if (xmt>et){
    cout << "Too low energy: " << et << " needed " << xmt << endl;
    return;
  }
// the parameter values are now accepted

// generate n massless momenta in infinite phase space
  for(int i=0; i<n;i++){
    double r1=xorshift64_rand(rng);
    double c=2.*r1-1.;
    double s=sqrt(1.-c*c);
    double f=twopi*xorshift64_rand(rng);
    r1=xorshift64_rand(rng);
    double r2=xorshift64_rand(rng);
    q[i][0]=-log(r1*r2);
    q[i][3]=q[i][0]*c;
    q[i][2]=q[i][0]*s*cos(f);
    q[i][1]=q[i][0]*s*sin(f);
  }
// calculate the parameters of the conformal transformation
  for (int i=0;i < 4;i++)
    r[i]=0.;
  for(int i=0;i < n;i++){
    for (int k=0; k<4; k++)
      r[k]=r[k]+q[i][k];
  }
  double rmas=sqrt(pow(r[0],2)-pow(r[3],2)-pow(r[2],2)-pow(r[1],2));
  for(int k=1;k < 4; k++)
    b[k-1]=-r[k]/rmas;
  double g=r[0]/rmas;
  double a=1./(1.+g);
  double x=et/rmas;

// transform the q's conformally into the p's
  for(int i=0; i< n;i++){
    double bq=b[0]*q[i][1]+b[1]*q[i][2]+b[2]*q[i][3];
    for (int k=1;k<4;k++)
      p[i][k]=x*(q[i][k]+b[k-1]*(q[i][0]+a*bq));
    p[i][0]=x*(g*q[i][0]+bq);
  }

// calculate weight and possible warnings
  wt=po2log;
  if(n!=2) wt=(2.*n-4.)*log(et)+z[n-1];
  if(wt<-180.){
    if(iwarn[0]<=5) cout << "Too small wt, risk for underflow: " << wt << endl;
    iwarn[0]=iwarn[0]+1;
  }
  if(wt> 174.){
    if(iwarn[1]<=5) cout << "Too large wt, risk for overflow: " << wt << endl;
    iwarn[1]=iwarn[1]+1;
  }

// return for weighted massless momenta
  if(nm==0){
// return log of weight
    for (int i=0;i < n;i++){
      for(int k=0;k <4;k++)
        momenta_out[i][k]=p[i][k];
    }
    return;
  }

// massive particles: rescale the momenta by a factor x
  double xmax=sqrt(1.-pow(xmt/et, 2));
  for(int i=0;i < n; i++){
    xm2[i]=pow(xm[i],2);
    p2[i]=pow(p[i][0],2);
  }
  int iter=0;
  x=xmax;
  double accu=et*acc;
  while(true){
    double f0=-et;
    double g0=0.;
    double x2=x*x;
    for(int i=0; i < n; i++){
      e[i]=sqrt(xm2[i]+x2*p2[i]);
      f0=f0+e[i];
      g0=g0+p2[i]/e[i];
    }
    if(abs(f0)<=accu) break;
    iter=iter+1;
    if(iter>itmax){
      cout << "Too many iterations without desired accuracy: " << itmax << endl;
      break;
    }
    x=x-f0/(x*g0);
  }
  for(int i=0;i < n;i++){
    v[i]=x*p[i][0];
    for(int k=1;k < 4; k++)
      p[i][k]=x*p[i][k];
    p[i][0]=e[i];
  }

// calculate the mass-effect weight factor
  double wt2=1.;
  double wt3=0.;
  for(int i=0;i < n; i++){
    wt2=wt2*v[i]/e[i];
    wt3=wt3+pow(v[i],2)/e[i];
  }
  double wtm=(2.*n-3.)*log(x)+log(wt2/wt3*et);

// return for  weighted massive momenta
  wt=wt+wtm;
  if(wt<-180.){
    if(iwarn[2]<=5) cout << "Too small wt, risk for underflow: " << wt << endl;
    iwarn[2]=iwarn[2]+1;
  }
  if(wt> 174.){
    if(iwarn[3]<=5)  cout << "Too large wt, risk for overflow: " << wt << endl;
    iwarn[3]=iwarn[3]+1;
  }
// return log of weight
  for (int i=0;i < n;i++){
    for(int k=0;k <4;k++)
      momenta_out[i][k]=p[i][k];
  }
  return;
}


