// Driver for the ORIGINAL lpzrobots Sox (selforg/controller/sox.cpp). Same protocol as driver.cpp:
// modes "sox" (Sox::step over a sensor stream) and "learn" (one Sox::learn; prints A C S b h v_avg L).
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <selforg/matrix.h>
#define protected public
#define private public
#include <selforg/sox.h>
#undef protected
#undef private
using namespace matrix;

static void rd(Matrix& M, int r, int c){ M.set(r,c); for(int i=0;i<r;i++) for(int j=0;j<c;j++){double v; std::cin>>v; M.val(i,j)=v;} }
static void pr(const Matrix& M){ for(unsigned i=0;i<M.getM();i++) for(unsigned j=0;j<M.getN();j++) std::cout<<M.val(i,j)<<' '; std::cout<<'\n'; }

int main(){
  std::cout<<std::setprecision(17);
  std::string mode; int ns,nm,nconf; std::cin>>mode>>ns>>nm>>nconf;
  SoxConf conf = Sox::getDefaultConf();
  std::vector<std::pair<std::string,double>> kv;
  for(int i=0;i<nconf;i++){ std::string k; double v; std::cin>>k>>v; kv.push_back({k,v});
    if(k=="initFeedbackStrength") conf.initFeedbackStrength=v;
    else if(k=="useExtendedModel") conf.useExtendedModel=v!=0;
    else if(k=="steps4Averaging") conf.steps4Averaging=(int)v;
    else if(k=="steps4Delay") conf.steps4Delay=(int)v; }
  Sox sox(conf);
  for(auto& p: kv){ const std::string& k=p.first; double v=p.second;
    if(k=="epsA") sox.epsA=v; else if(k=="epsC") sox.epsC=v; else if(k=="creativity") sox.creativity=v;
    else if(k=="harmony") sox.harmony=v; else if(k=="causeaware") sox.causeaware=v; else if(k=="sense") sox.sense=v;
    else if(k=="damping") sox.damping=v; else if(k=="loga") sox.loga=v!=0;
    else if(k=="initFeedbackStrength"||k=="useExtendedModel"||k=="steps4Averaging"||k=="steps4Delay"){}
    else { std::cerr<<"unknown key "<<k<<"\n"; return 2; } }
  sox.init(ns,nm);
  const int bs = Sox::buffersize;
  if(mode=="sox"){
    int T; std::cin>>T; std::vector<double> x(ns), y(nm);
    for(int s=0;s<T;s++){ for(auto& e:x) std::cin>>e; sox.step(x.data(),ns,y.data(),nm);
      for(double e:y) std::cout<<e<<' '; std::cout<<'\n'; }
  } else if(mode=="learn"){
    int t; std::cin>>t; sox.t=t;
    rd(sox.A,ns,nm); rd(sox.C,nm,ns); rd(sox.S,ns,ns); rd(sox.b,ns,1); rd(sox.h,nm,1); rd(sox.v_avg,ns,1);
    for(int k=0;k<bs;k++) rd(sox.x_buffer[k],ns,1);
    for(int k=0;k<bs;k++) rd(sox.y_buffer[k],nm,1);
    sox.learn();
    pr(sox.A); pr(sox.C); pr(sox.S); pr(sox.b); pr(sox.h); pr(sox.v_avg); pr(sox.L);
  } else { std::cerr<<"unknown mode\n"; return 2; }
  return 0;
}
