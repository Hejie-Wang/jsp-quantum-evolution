// Classical reference heuristic, NOT quantum or the decomposed QUBO algorithm.
// Usage: ./tabu_baseline input.txt seconds seed target_lb output_order.txt [insert]
// Critical block adjacent swaps (default) or insertions; integer DAG evaluation.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>
using namespace std;
using Order=vector<vector<int>>;
struct Eval { int value; vector<int> head,tail; };
int n,m,N;
vector<int> p,mac,js;
mt19937 rng;

Eval evaluate(const Order& ord,bool tails=false){
    vector<int> ms(N,-1),deg(N,0),h(N,0),top;
    top.reserve(N);
    for(int u=0;u<N;u++) if(js[u]>=0)deg[js[u]]++;
    for(const auto& row:ord)for(int k=1;k<n;k++)ms[row[k-1]]=row[k],deg[row[k]]++;
    for(int u=0;u<N;u++)if(!deg[u])top.push_back(u);
    int makespan=0;
    for(size_t i=0;i<top.size();i++){
        int u=top[i]; makespan=max(makespan,h[u]+p[u]);
        for(int v:{js[u],ms[u]})if(v>=0){
            h[v]=max(h[v],h[u]+p[u]);
            if(--deg[v]==0)top.push_back(v);
        }
    }
    if(int(top.size())!=N)return {1000000000,{}, {}};
    vector<int> tail;
    if(tails){
        tail.assign(N,0);
        for(auto it=top.rbegin();it!=top.rend();it++){
            int u=*it;
            for(int v:{js[u],ms[u]})if(v>=0)tail[u]=max(tail[u],p[v]+tail[v]);
        }
    }
    return {makespan,move(h),move(tail)};
}

Order initial(){
    vector<int> next(n,0),jr(n,0),mr(m,0),remaining(n,0);
    for(int j=0;j<n;j++)for(int k=0;k<m;k++)remaining[j]+=p[j*m+k];
    Order ord(m);
    for(int step=0;step<N;step++){
        int earliest=1000000000,km=-1;
        for(int j=0;j<n;j++)if(next[j]<m){
            int v=j*m+next[j],end=max(jr[j],mr[mac[v]])+p[v];
            if(end<earliest)earliest=end,km=mac[v];
        }
        vector<int> conflict;
        for(int j=0;j<n;j++)if(next[j]<m){
            int v=j*m+next[j];
            if(mac[v]==km && max(jr[j],mr[km])<earliest)conflict.push_back(j);
        }
        int selected=conflict[0];double best=-1;
        for(int j:conflict){
            double score=remaining[j]*(0.75+double(rng()%10000)/10000.0);
            if(score>best)best=score,selected=j;
        }
        int v=selected*m+next[selected]++;
        jr[selected]=mr[km]=max(jr[selected],mr[km])+p[v];
        remaining[selected]-=p[v];ord[km].push_back(v);
    }
    return ord;
}

void insert_move(vector<int>& row,int a,int b){
    if(a<b)rotate(row.begin()+a,row.begin()+a+1,row.begin()+b+1);
    else rotate(row.begin()+b,row.begin()+a,row.begin()+a+1);
}

int main(int argc,char**argv){
    if(argc!=6 && argc!=7){cerr<<"usage: input seconds seed target output [insert]\n";return 2;}
    bool insertion=(argc==7 && string(argv[6])=="insert");
    ifstream f(argv[1]);string line;vector<vector<int>> rows;
    while(getline(f,line)){istringstream s(line);vector<int> r;int a;while(s>>a)r.push_back(a);if(!r.empty())rows.push_back(r);}
    n=rows.size()/2;m=rows[0].size();N=n*m;
    p.resize(N);mac.resize(N);js.assign(N,-1);
    for(int j=0;j<n;j++)for(int k=0;k<m;k++){
        int u=j*m+k;p[u]=rows[j][k];mac[u]=rows[j+n][k]-1;if(k<m-1)js[u]=u+1;
    }
    double budget=stod(argv[2]);rng.seed(stoul(argv[3]));int target=stoi(argv[4]);
    auto begin=chrono::steady_clock::now();
    auto elapsed=[&](){return chrono::duration<double>(chrono::steady_clock::now()-begin).count();};
    Order current=initial(),best=current;
    int bestval=evaluate(best).value,iteration=0,lastbest=0,restarts=0;
    vector<int> tabu(N*N,0);
    cerr<<"initial "<<bestval<<"\n";
    while(elapsed()<budget && bestval>target){
        iteration++;
        auto e=evaluate(current,true);
        vector<tuple<int,int,int>> moves;
        for(int k=0;k<m;k++){
            for(int a=0;a<n-1;){
                auto critical=[&](int i){int u=current[k][i],v=current[k][i+1];return e.head[u]+p[u]+p[v]+e.tail[v]==e.value;};
                if(!critical(a)){a++;continue;}
                int b=a;while(b<n-2 && critical(b+1))b++;
                int last=b+1;
                if(insertion){
                    for(int c=a+1;c<=last;c++){moves.push_back({k,a,c});moves.push_back({k,c,a});}
                    for(int c=a;c<last;c++){moves.push_back({k,last,c});moves.push_back({k,c,last});}
                } else {
                    moves.push_back({k,a,a+1});if(b>a)moves.push_back({k,b,b+1});
                }
                a=b+1;
            }
        }
        shuffle(moves.begin(),moves.end(),rng);
        int chosen=-1,score=1000000000, fallback=-1, fallback_score=1000000000;
        for(int i=0;i<int(moves.size());i++){
            auto [k,a,b]=moves[i];int u=current[k][min(a,b)],v=current[k][max(a,b)];
            insert_move(current[k],a,b);int val=evaluate(current).value;insert_move(current[k],b,a);
            if(val>=1000000000)continue;
            if(val<fallback_score)fallback_score=val,fallback=i;
            if(tabu[v*N+u]>iteration && val>=bestval)continue;
            if(val<score)score=val,chosen=i;
        }
        if(chosen<0 && insertion){chosen=fallback;score=fallback_score;}
        if(chosen<0 || iteration-lastbest>2500){
            current=initial();fill(tabu.begin(),tabu.end(),0);lastbest=iteration;restarts++;continue;
        }
        auto [k,a,b]=moves[chosen];int u=current[k][min(a,b)],v=current[k][max(a,b)];
        tabu[u*N+v]=iteration+7+rng()%11;insert_move(current[k],a,b);
        if(score<bestval){
            bestval=score;best=current;lastbest=iteration;
            cerr<<"best "<<bestval<<" iteration "<<iteration<<" seconds "<<elapsed()<<"\n";
        }
    }
    ofstream out(argv[5]);for(auto& row:best){for(int u:row)out<<u<<' ';out<<'\n';}
    cout<<"{\"makespan\":"<<bestval<<",\"iterations\":"<<iteration<<",\"restarts\":"<<restarts
        <<",\"seconds\":"<<elapsed()<<",\"seed\":"<<argv[3]<<",\"target_lower_bound\":"<<target<<"}\n";
}
