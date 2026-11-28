// Fixed-path pair simulation with shared-window and independent-endpoint scoring.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr double dt = .1, cap = 15, acceleration = 2, deceleration = 4;
constexpr int samples = 121, replan = 4;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr std::array<double, 8> waits{0,.5,1,1.5,2,3,4,5};
constexpr std::array<double, 5> speeds{0,7,10,13,15};

struct Pose { double x,y,c,s; };
struct Path {
    const double *data; // arc length, x, y, unwrapped angle
    int n;
    double entry, exit;
    double end() const { return data[4*(n-1)]; }
    Pose at(double position) const {
        int lo=0, hi=n-1;
        while (lo+1<hi) {
            int mid=(lo+hi)/2;
            if (data[4*mid]<=position) lo=mid; else hi=mid;
        }
        const auto *a=data+4*lo, *b=data+4*hi;
        const double f=std::clamp((position-a[0])/(b[0]-a[0]),0.,1.);
        const double theta=a[3]+f*(b[3]-a[3]);
        return {a[1]+f*(b[1]-a[1]),a[2]+f*(b[2]-a[2]),std::cos(theta),std::sin(theta)};
    }
};
struct Size { double l,w; };
struct State { double p,v; };
struct Offer { double wait,target,ramp; };
struct Profile {
    std::array<State,samples> trajectory;
    mutable std::array<Pose,(samples+1)/2> poses{};
    mutable int poses_until=0;
    double finish, arrival, cost, leave;
    Offer offer;
};
struct Metrics {
    double minimum=inf, pa=0, pb=0, finish=inf;
    double finish_a=inf,finish_b=inf;
    double exposure=0, elapsed=0, first_contact=inf;
    int overlaps=0, near=0, initial=0, ever=0, fallback=0, replans=0, refinements=0, stalled=0;
};
struct Decision {
    int tick, steps, feasible_pairs;
    bool fallback;
    std::array<State,2> before, after;
    std::array<Offer,2> offers;
};
using Pair = std::array<int,2>;
using Banks = std::array<std::vector<Profile>,2>;

double gap_axis(const Pose&a,const Pose&b,const Size&sa,const Size&sb) {
    const double dx=b.x-a.x,dy=b.y-a.y;
    if (std::abs(dx)>sa.l+sa.w+sb.l+sb.w+.01 || std::abs(dy)>sa.l+sa.w+sb.l+sb.w+.01)
        return 1;
    const double cos=std::abs(a.c*b.c+a.s*b.s),sin=std::abs(a.c*b.s-a.s*b.c);
    return std::max({std::abs(dx*a.c+dy*a.s)-sa.l-sb.l*cos-sb.w*sin,
                     std::abs(-dx*a.s+dy*a.c)-sa.w-sb.l*sin-sb.w*cos,
                     std::abs(dx*b.c+dy*b.s)-sb.l-sa.l*cos-sa.w*sin,
                     std::abs(-dx*b.s+dy*b.c)-sb.w-sa.l*sin-sa.w*cos});
}
double point_box_squared(double x,double y,const Pose&b,const Size&sb) {
    const double dx=x-b.x,dy=y-b.y;
    const double f=std::max(0.,std::abs(dx*b.c+dy*b.s)-sb.l);
    const double s=std::max(0.,std::abs(-dx*b.s+dy*b.c)-sb.w);
    return f*f+s*s;
}
double box_gap(const Pose&a,const Pose&b,const Size&sa,const Size&sb,double previous) {
    const double radius=std::hypot(sa.l,sa.w)+std::hypot(sb.l,sb.w);
    if (std::abs(a.x-b.x)>radius+previous || std::abs(a.y-b.y)>radius+previous) return previous;
    if (gap_axis(a,b,sa,sb)<=1e-9) return 0.;
    double result=previous*previous;
    for (int f:{-1,1}) for (int s:{-1,1}) {
        result=std::min(result,point_box_squared(a.x+f*sa.l*a.c-s*sa.w*a.s,a.y+f*sa.l*a.s+s*sa.w*a.c,b,sb));
        result=std::min(result,point_box_squared(b.x+f*sb.l*b.c-s*sb.w*b.s,b.y+f*sb.l*b.s+s*sb.w*b.c,a,sa));
    }
    return std::sqrt(result);
}
State advance(State state,double target,double end,double ramp=1) {
    const double next=std::max(0.,state.v+dt*std::clamp((target-state.v)/dt,-deceleration*ramp,acceleration*ramp));
    return {std::min(end,state.p+dt*(state.v+next)/2),next};
}
double interpolated_hit(double level,State before,State after,int i) {
    return dt*(i-1+(level-before.p)/(after.p-before.p));
}
Profile profile(const Path&path,State state,Offer offer,bool endpoint_exit=false) {
    Profile out;
    out.offer=offer;
    out.arrival=state.p>=path.end()?0:inf;
    out.leave=state.p>=path.exit?0:inf;
    out.trajectory[0]=state;
    double effort=0;
    bool ramp_started=false;
    double ramp_fraction=offer.ramp;
    const int hold_steps=int(std::nearbyint(offer.wait/dt));
    // The discrete grid uses a literal time comparison; the continuous seeds
    // round waits to their 0.1 s grid as in ContinuousOfferSearch.
    for (int k=1;k<samples;k++) {
        const bool hold=(offer.ramp<0?(k-1)*dt<offer.wait:k-1<hold_steps) && state.p<path.entry;
        if (!hold&&!ramp_started) {
            ramp_fraction=state.v>cap||offer.ramp<0?1:offer.ramp;
            ramp_started=true;
        }
        const double target=hold?std::min({state.v,cap,std::sqrt(2*deceleration*std::max(0.,path.entry-.7-state.p))}):offer.target;
        State next=endpoint_exit&&state.p>=path.end()?state:
            advance(state,target,endpoint_exit?inf:path.end(),hold?1:ramp_fraction);
        if (state.p<path.end()) effort+=(next.v-state.v)*(next.v-state.v)/dt;
        if (!std::isfinite(out.arrival)&&next.p>=path.end()) out.arrival=interpolated_hit(path.end(),state,next,k);
        if (!std::isfinite(out.leave)&&next.p>=path.exit) out.leave=interpolated_hit(path.exit,state,next,k);
        out.trajectory[k]=next;
        state=next;
    }
    out.finish=std::isfinite(out.arrival)?out.arrival:12+std::max(0.,path.end()-state.p)/2;
    out.cost=out.finish+.02*effort;
    return out;
}
double forecast_position(const Profile& offer,double time) {
    const double station=time/dt;
    const int lower=std::clamp(int(station),0,samples-2);
    return offer.trajectory[lower].p+(station-lower)*(offer.trajectory[lower+1].p-offer.trajectory[lower].p);
}
bool safe(const Profile&a,const Profile&b,const Path*paths,const Size*size,int checks,bool endpoint_exit=false) {
    const double departure=endpoint_exit?std::min(a.arrival,b.arrival):inf;
    for (int i=1;i<=checks;i++) {
        if (i*.2>=departure) break;
        if (i>a.poses_until) {a.poses[i]=paths[0].at(a.trajectory[2*i].p);a.poses_until=i;}
        if (i>b.poses_until) {b.poses[i]=paths[1].at(b.trajectory[2*i].p);b.poses_until=i;}
        if (gap_axis(a.poses[i],b.poses[i],size[0],size[1])<=1e-6) return false;
    }
    if (departure<=checks*.2)
        return gap_axis(paths[0].at(forecast_position(a,departure)),paths[1].at(forecast_position(b,departure)),
                        size[0],size[1])>1e-6;
    return true;
}
double ttc(const Path* paths,const State* state,const Size* sizes) {
    for (int i=0;i<=50;i++) {
        double a=state[0].p+i*dt*state[0].v,b=state[1].p+i*dt*state[1].v;
        if (a>paths[0].end()||b>paths[1].end()) break;
        if (gap_axis(paths[0].at(a),paths[1].at(b),sizes[0],sizes[1])<=0) return i*dt;
    }
    return inf;
}
double eta(double entry,State s) {
    if (s.p>=entry) return 0;
    return s.v>0?(entry-s.p)/s.v:inf;
}
Pair independent(const Banks&banks,bool mpc) {
    Pair ids{0,0};
    for (int j=0;j<2;j++) for (int k=1;k<int(banks[j].size());k++)
        if ((mpc?banks[j][k].cost:banks[j][k].finish)<(mpc?banks[j][ids[j]].cost:banks[j][ids[j]].finish))
            ids[j]=k;
    return ids;
}
void build_bank(std::vector<Profile>&bank,const Path&path,State state,bool continuous,bool endpoint_exit,
                bool include_stop=false) {
    bank.clear();
    for (auto wait:waits) {
        for (int s=(continuous||include_stop)?0:1;s<=(continuous?4:3);s++)
            bank.push_back(profile(path,state,{wait,speeds[s],continuous?1.:-1.},endpoint_exit));
        if (state.p>=path.entry) break;
    }
}
Pair discrete_selection(const Banks& b,const std::vector<Pair>& valid,double delta,int first,bool mpc) {
    if (mpc) return *std::min_element(valid.begin(),valid.end(),[&](Pair a,Pair c) {
        return b[0][a[0]].cost+b[1][a[1]].cost < b[0][c[0]].cost+b[1][c[1]].cost;
    });
    double minima[2]{inf,inf};
    for (auto p:valid) for(int j=0;j<2;j++) minima[j]=std::min(minima[j],b[j][p[j]].finish);
    double continuation[2]{0,0};
    Pair selected=valid[0];
    for (int actor:{first,1-first,first}) {
        const int other=1-actor;
        double best=-1,total=inf;
        Pair offer=selected;
        for (auto p:valid) {
            if (std::exp(-(b[other][p[other]].finish-minima[other])/2)<delta*continuation[other]-1e-12) continue;
            double value=std::exp(-(b[actor][p[actor]].finish-minima[actor])/2);
            double sum=b[0][p[0]].finish+b[1][p[1]].finish;
            if (value>best||(value==best&&sum<total)) {best=value;total=sum;offer=p;}
        }
        if (best>=delta*continuation[actor]-1e-12) {
            selected=offer;
            for (int j=0;j<2;j++) continuation[j]=std::exp(-(b[j][offer[j]].finish-minima[j])/2);
        } else for (double &x:continuation) x*=delta;
    }
    return selected;
}
bool preferred(const std::array<Profile,2>&a,const std::array<Profile,2>&b,int actor) {
    return a[actor].finish<b[actor].finish ||
           (a[actor].finish==b[actor].finish && a[0].finish+a[1].finish < b[0].finish+b[1].finish);
}
std::array<Profile,2> refine(std::array<Profile,2> candidate,const Path*paths,const State*state,
                            const Size*sizes,int checks,int actor,double responder_limit,Metrics&metrics,bool endpoint_exit) {
    // Refine real-valued target speeds and acceleration fractions from the feasible
    // seed using bounded coordinate search. Accept only feasible improvements.
    for (double scale:{1.,.5,.25,.125}) {
        for (int j:{actor,1-actor}) {
            if (j!=actor && !std::isfinite(responder_limit)) continue;
            for (int coordinate:{0,1}) for (double sign:{-1.,1.}) {
                Offer choice=candidate[j].offer;
                double& value=coordinate==0?choice.target:choice.ramp;
                const double original=value;
                const double endpoint=std::clamp(value+sign*scale*(coordinate==0?3:.5),
                                                0.,coordinate==0?cap:1.);
                if (original==endpoint) continue;
                for (double fraction:{1.,.6180339887498949,.3819660112501051}) {
                    value=original+fraction*(endpoint-original);
                    auto updated=candidate;
                    updated[j]=profile(paths[j],state[j],choice,endpoint_exit);
                    ++metrics.refinements;
                    if (!preferred(updated,candidate,actor)) break;
                    if (updated[1-actor].finish<=responder_limit+1e-8 &&
                        safe(updated[0],updated[1],paths,sizes,checks,endpoint_exit)) {
                        candidate=updated;
                        break;
                    }
                }
            }
        }
    }
    return candidate;
}
std::array<Profile,2> continuous_selection(const Banks&b,const std::vector<Pair>&valid,const Path* paths,
        const State*state,const Size*sizes,int checks,double delta,int first,Metrics&metrics,bool endpoint_exit) {
    std::array<Profile,2> result{b[0][valid[0][0]],b[1][valid[0][1]]};
    bool started=false;
    for (int actor:{first,1-first,first}) {
        double limit=started&&delta>0?result[1-actor].finish-2*std::log(delta):inf;
        std::array<Pair,3> seeds{Pair{-1,-1},Pair{-1,-1},Pair{-1,-1}};
        for (auto pair:valid) {
            const auto &a=b[0][pair[0]],&c=b[1][pair[1]];
            if (b[1-actor][pair[1-actor]].finish>limit+1e-8) continue;
            int order=state[0].p>=paths[0].exit||state[1].p>=paths[1].exit||
                       (!std::isfinite(a.leave)&&!std::isfinite(c.leave))?2:(a.leave<=c.leave?0:1);
            Pair& old=seeds[order];
            if (old[0]<0||b[actor][pair[actor]].finish<b[actor][old[actor]].finish
                ||(b[actor][pair[actor]].finish==b[actor][old[actor]].finish&&
                   a.finish+c.finish<b[0][old[0]].finish+b[1][old[1]].finish)) old=pair;
        }
        for (auto seed:seeds) {
            if (seed[0]<0) continue;
            std::array<Profile,2> proposal{b[0][seed[0]],b[1][seed[1]]};
            proposal=refine(proposal,paths,state,sizes,checks,actor,limit,metrics,endpoint_exit);
            if (!started||preferred(proposal,result,actor)) {result=proposal;started=true;}
        }
    }
    return result;
}
double physical_horizon(const Path*paths,const double*speeds,double available) {
    double horizon=available;
    for(int j=0;j<2;j++) {
        State v{0,speeds[j]};
        for(int step=1;(step-1)*dt<horizon;step++) {
            State next=advance(v,cap,inf);
            if (next.p>=paths[j].end()) {horizon=std::min(horizon,interpolated_hit(paths[j].end(),v,next,step));break;}
            v=next;
        }
    }
    return horizon;
}
Metrics assess(const Path*paths,const Size*sizes,const double*times,const double*progress,int n,double horizon,
               const double* goals=nullptr,bool endpoint_exit=false,bool fail_on_contact=false) {
    Metrics m;
    m.elapsed=horizon;
    if(goals) {
        double *hit[2]{&m.finish_a,&m.finish_b};
        for(int j=0;j<2;j++) {
            if (!std::isfinite(goals[j])) continue;
            if (progress[j]>=goals[j]) {*hit[j]=times[0];continue;}
            for(int i=1;i<n&&times[i-1]<=horizon;i++) {
                double a=progress[2*i-2+j],b=progress[2*i+j];
                if(b>=goals[j]&&b>a) {
                    double arrival=times[i-1]+(times[i]-times[i-1])*(goals[j]-a)/(b-a);
                    if(arrival<=horizon+1e-9) *hit[j]=arrival;
                    break;
                }
            }
        }
        m.finish=std::max(m.finish_a,m.finish_b);
    }
    if (endpoint_exit) {
        m.elapsed=std::min(horizon,m.finish);
        horizon=std::min({horizon,m.finish_a,m.finish_b});
    }
    m.exposure=horizon;
    int index=0;
    bool was_contact=false;
    auto observe=[&](double clock,bool first) {
        while(index+1<n-1&&times[index+1]<clock) ++index;
        const int next=std::min(index+1,n-1);
        const double weight=n==1?0.:std::clamp((clock-times[index])/(times[next]-times[index]),0.,1.);
        const double a=progress[2*index]+weight*(progress[2*next]-progress[2*index]);
        const double b=progress[2*index+1]+weight*(progress[2*next+1]-progress[2*index+1]);
        const auto pa=paths[0].at(a),pb=paths[1].at(b);
        // Use a fresh box distance for labels; short circuit the distance only once
        // the available bounds rule out both a near miss and a new global minimum.
        double d=box_gap(pa,pb,sizes[0],sizes[1],std::max(1.00001,m.minimum));
        m.minimum=std::min(m.minimum,d);
        const bool contact=d<=1e-9;
        if(contact && !m.ever) m.first_contact=clock;
        if(first) m.initial=contact;
        else if(contact&&!was_contact) m.overlaps=1;
        m.ever|=contact;
        m.near|=!contact&&d<=1.;
        was_contact=contact;
        m.pa=a;m.pb=b;
    };
    observe(0,true);
    for(int i=1;i*.02<horizon&&(!fail_on_contact||!m.ever);i++) observe(i*.02,false);
    if(horizon>0&&(!fail_on_contact||!m.ever)) observe(horizon,false);
    if(fail_on_contact&&m.ever) {
        m.exposure=m.elapsed=m.first_contact;
        m.finish_a=m.finish_b=m.finish=inf;
    }
    // Keep the conventional mutually exclusive near-miss and contact rates.
    m.near &= !m.ever;
    return m;
}

Profile solo_mpc(const Path& path,State state) {
    auto best=profile(path,state,{0,7,-1},true);
    for (double speed:{10.,13.}) {
        auto candidate=profile(path,state,{0,speed,-1},true);
        if(candidate.cost<best.cost) best=candidate;
    }
    return best;
}
bool same_state(const State* a,const State* b) {
    return a[0].p==b[0].p&&a[0].v==b[0].v&&a[1].p==b[1].p&&a[1].v==b[1].v;
}

bool contact_at(const Path* paths,const Size* sizes,double a,double b) {
    return gap_axis(paths[0].at(a),paths[1].at(b),sizes[0],sizes[1])<=1e-9;
}

bool executed_contact(const Path* paths,const Size* sizes,const State* before,const State* after,int tick) {
    if (before[0].p>=paths[0].end()||before[1].p>=paths[1].end()) return false;
    const double start=(tick-1)*dt,end=tick*dt;
    double active_until=end;
    for (int j=0;j<2;j++) if (after[j].p>=paths[j].end()) {
        active_until=std::min(active_until,start+(end-start)*(paths[j].end()-before[j].p)/(after[j].p-before[j].p));
    }
    auto contact=[&](double time) {
        const double f=std::clamp((time-start)/(end-start),0.,1.);
        return contact_at(paths,sizes,before[0].p+f*(after[0].p-before[0].p),
                          before[1].p+f*(after[1].p-before[1].p));
    };
    for (int sample=(tick-1)*5+1;sample*.02<active_until;sample++)
        if (contact(sample*.02)) return true;
    return contact(active_until);
}

Metrics simulate(const Path*paths,const Size*sizes,const double*initial,double horizon,int arm,const double* goals,
                 bool endpoint_exit=false,std::vector<Decision>* trace=nullptr,
                 bool fail_on_contact=false,bool unfiltered=false) {
    const bool continuous=arm==0||arm==2, use_ttc=arm==0||arm==1, mpc=arm==4;
    State state[2]{{0,initial[0]},{0,initial[1]}};
    std::vector<double> times{0}, progress{0,0};
    Metrics work;
    Banks banks;
    for(auto& bank:banks) bank.reserve(40);
    const int steps=endpoint_exit?0:std::max(1,int(std::ceil(horizon/dt)));
    bool detected=fail_on_contact&&contact_at(paths,sizes,0,0);
    for(int tick=0;!detected&&(endpoint_exit||tick<steps);) {
        const int start_tick=tick;
        const bool done_a=state[0].p>=paths[0].end(),done_b=state[1].p>=paths[1].end();
        if (endpoint_exit&&done_a&&done_b) break;
        const int execute=endpoint_exit?replan:std::min(replan,steps-tick);
        const int checks=endpoint_exit?60:std::clamp(int(std::ceil((horizon-tick*dt)/(.2))),1,60);
        const State previous[2]{state[0],state[1]};
        ++work.replans;
        std::array<Profile,2> chosen{
            profile(paths[0],state[0],{0,continuous?15.:13.,continuous?1.:-1.},endpoint_exit),
            profile(paths[1],state[1],{0,continuous?15.:13.,continuous?1.:-1.},endpoint_exit)};
        const bool alone=endpoint_exit&&(done_a||done_b);
        if (alone&&mpc) {int j=done_a?1:0;chosen[j]=solo_mpc(paths[j],state[j]);}
        // With own arrival time as the utility, the individually fastest pair
        // is jointly optimal whenever the game has no feasibility filter.
        bool have=unfiltered||alone||(!mpc&&safe(chosen[0],chosen[1],paths,sizes,checks,endpoint_exit));
        int feasible_pairs=-1; // Shortcuts do not enumerate every pair.
        if (!have) {
            for (int j=0;j<2;j++) build_bank(banks[j],paths[j],state[j],continuous,endpoint_exit);
            Pair optimal=independent(banks,mpc);
            if (safe(banks[0][optimal[0]],banks[1][optimal[1]],paths,sizes,checks,endpoint_exit)) {
                chosen={banks[0][optimal[0]],banks[1][optimal[1]]};
                have=true;
            } else {
                std::vector<Pair> valid;
                for(int i=0;i<int(banks[0].size());i++) for(int j=0;j<int(banks[1].size());j++)
                    if(safe(banks[0][i],banks[1][j],paths,sizes,checks,endpoint_exit)) valid.push_back({i,j});
                feasible_pairs=int(valid.size());
                if(!valid.empty()) {
                    const double delta=use_ttc?1-std::exp(-.3*ttc(paths,state,sizes)):.8;
                    const int first=eta(paths[1].entry,state[1])<eta(paths[0].entry,state[0])?1:0;
                    if(continuous) chosen=continuous_selection(banks,valid,paths,state,sizes,checks,delta,first,work,endpoint_exit);
                    else {Pair p=discrete_selection(banks,valid,delta,first,mpc);chosen={banks[0][p[0]],banks[1][p[1]]};}
                    have=true;
                }
            }
        }
        work.fallback+=!have;
        for(int step=1;step<=execute;step++) {
            const State before[2]{state[0],state[1]};
            for(int j=0;j<2;j++) if(!endpoint_exit||state[j].p<paths[j].end())
                state[j]=have?chosen[j].trajectory[step]:advance(state[j],0,endpoint_exit?inf:paths[j].end());
            times.push_back(++tick*dt);
            progress.push_back(state[0].p);progress.push_back(state[1].p);
            if(fail_on_contact&&executed_contact(paths,sizes,before,state,tick)) {detected=true;break;}
            // Begin independent control at the first dynamics tick following an exit.
            if(endpoint_exit&&((!done_a&&state[0].p>=paths[0].end())||(!done_b&&state[1].p>=paths[1].end()))) break;
        }
        if(trace) trace->push_back({start_tick,tick-start_tick,feasible_pairs,!have,
                                    {previous[0],previous[1]},{state[0],state[1]},
                                    {chosen[0].offer,chosen[1].offer}});
        if(endpoint_exit&&!detected&&same_state(previous,state)) {work.stalled=1;break;}
    }
    auto result=assess(paths,sizes,times.data(),progress.data(),int(times.size()),
                       endpoint_exit?times.back():horizon,goals,endpoint_exit,fail_on_contact);
    result.replans=work.replans;result.fallback=work.fallback;result.refinements=work.refinements;
    result.stalled=work.stalled&&(!fail_on_contact||!result.ever);
    return result;
}
Metrics fixed_release(const Path*paths,const Size*sizes,const double*initial,double horizon,const double* goals,
                      bool endpoint_exit=false,bool fail_on_contact=false) {
    State states[2]{{0,initial[0]},{0,initial[1]}};
    const int priority=paths[1].entry/std::max(.5,initial[1]) <
                       paths[0].entry/std::max(.5,initial[0]) ? 1:0;
    const int yielding=1-priority;
    const double release=(paths[priority].entry+paths[priority].exit)/2+5;
    std::vector<double> times{0},progress{0,0};
    const int steps=endpoint_exit?0:std::max(1,int(std::ceil(horizon/dt)));
    bool stalled=false;
    bool detected=fail_on_contact&&contact_at(paths,sizes,0,0);
    for(int i=1;!detected&&(endpoint_exit||i<=steps);i++) {
        if(endpoint_exit&&states[0].p>=paths[0].end()&&states[1].p>=paths[1].end()) break;
        const State previous[2]{states[0],states[1]};
        double targets[2]{10,10};
        if(states[priority].p<release-1e-6&&(!endpoint_exit||states[priority].p<paths[priority].end()))
            targets[yielding]=std::min({cap,states[yielding].v,std::sqrt(2*deceleration*
                                       std::max(0.,paths[yielding].entry-.7-states[yielding].p))});
        for(int j=0;j<2;j++) if(!endpoint_exit||states[j].p<paths[j].end())
            states[j]=advance(states[j],targets[j],endpoint_exit?inf:paths[j].end());
        times.push_back(i*dt);progress.push_back(states[0].p);progress.push_back(states[1].p);
        if(fail_on_contact&&executed_contact(paths,sizes,previous,states,i)) {detected=true;break;}
        if(endpoint_exit&&same_state(previous,states)) {stalled=true;break;}
    }
    auto result=assess(paths,sizes,times.data(),progress.data(),int(times.size()),
                       endpoint_exit?times.back():horizon,goals,endpoint_exit,fail_on_contact);
    result.stalled=stalled&&(!fail_on_contact||!result.ever);
    return result;
}
void serialize(double *dst,const Metrics&m) {
    const double columns[]{m.minimum,m.pa,m.pb,double(m.overlaps),double(m.near),double(m.initial),
                           double(m.ever),double(m.replans),double(m.fallback),double(m.refinements),
                           m.finish_a,m.finish_b,m.finish};
    std::copy(std::begin(columns),std::end(columns),dst);
}
}

extern "C" __declspec(dllexport) int evaluate_window(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *entry,const double *exit,const double *dimensions,const double *initial,
        const double *times,const double *progress,int n,const double *release,const double *goals,double *out) {
    if (n<2||count_a<2||count_b<2) return 1;
    Path paths[2]{{path_a,count_a,entry[0],exit[0]},{path_b,count_b,entry[1],exit[1]}};
    Size sizes[2]{{dimensions[0]/2,dimensions[1]/2},{dimensions[2]/2,dimensions[3]/2}};
    double horizon=physical_horizon(paths,initial,times[n-1]);
    out[0]=horizon;
    for (int arm=0;arm<6;arm++) {
        Metrics m=arm==5?assess(paths,sizes,times,progress,n,horizon,goals):simulate(paths,sizes,initial,horizon,arm,goals);
        serialize(out+1+arm*13,m);
    }
    if(std::isfinite(release[0])) {
        for(int j=0;j<2;j++) {paths[j].entry=release[j]; paths[j].exit=release[2+j];}
        serialize(out+79,fixed_release(paths,sizes,initial,horizon,goals));
    }
    return 0;
}

extern "C" __declspec(dllexport) void score_window(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *dimensions,const double *times,const double *progress,int n,double horizon,double *out) {
    Path paths[2]{{path_a,count_a,0,0},{path_b,count_b,0,0}};
    Size sizes[2]{{dimensions[0]/2,dimensions[1]/2},{dimensions[2]/2,dimensions[3]/2}};
    serialize(out,assess(paths,sizes,times,progress,n,horizon));
}

static int evaluate_terminal_case(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *entry,const double *exit,const double *dimensions,const double *initial,
        const double *times,const double *progress,int n,const double *release,double *out,bool unfiltered) {
    if (n<2||count_a<2||count_b<2) return 1;
    Path paths[2]{{path_a,count_a,entry[0],exit[0]},{path_b,count_b,entry[1],exit[1]}};
    Size sizes[2]{{dimensions[0]/2,dimensions[1]/2},{dimensions[2]/2,dimensions[3]/2}};
    const double goals[2]{paths[0].end(),paths[1].end()};
    const bool include_release=std::isfinite(release[0]);
    for (int arm=0;arm<6+int(include_release);arm++) {
        Metrics m;
        if(arm==5) m=assess(paths,sizes,times,progress,n,times[n-1],goals,true,unfiltered);
        else if(arm==6) {
            for(int j=0;j<2;j++) {paths[j].entry=release[j]; paths[j].exit=release[2+j];}
            m=fixed_release(paths,sizes,initial,0,goals,true,unfiltered);
        } else m=simulate(paths,sizes,initial,0,arm,goals,true,nullptr,unfiltered,unfiltered&&arm<4);
        double* slot=out+arm*(unfiltered?17:16);
        serialize(slot,m);
        slot[13]=m.exposure;slot[14]=m.elapsed;slot[15]=m.stalled;
        if(unfiltered) slot[16]=m.first_contact;
    }
    return 0;
}

extern "C" __declspec(dllexport) int evaluate_endpoints(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *entry,const double *exit,const double *dimensions,const double *initial,
        const double *times,const double *progress,int n,const double *release,double *out) {
    return evaluate_terminal_case(path_a,count_a,path_b,count_b,entry,exit,dimensions,initial,
                                  times,progress,n,release,out,false);
}

extern "C" __declspec(dllexport) int evaluate_unfiltered_endpoints(const double *path_a,int count_a,
        const double *path_b,int count_b,const double *entry,const double *exit,const double *dimensions,
        const double *initial,const double *times,const double *progress,int n,const double *release,double *out) {
    return evaluate_terminal_case(path_a,count_a,path_b,count_b,entry,exit,dimensions,initial,
                                  times,progress,n,release,out,true);
}

extern "C" __declspec(dllexport) void score_endpoints(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *dimensions,const double *times,const double *progress,int n,double *out) {
    Path paths[2]{{path_a,count_a,0,0},{path_b,count_b,0,0}};
    Size sizes[2]{{dimensions[0]/2,dimensions[1]/2},{dimensions[2]/2,dimensions[3]/2}};
    const double goals[2]{paths[0].end(),paths[1].end()};
    const auto m=assess(paths,sizes,times,progress,n,times[n-1],goals,true);
    serialize(out,m);
    out[13]=m.exposure;out[14]=m.elapsed;out[15]=m.stalled;
}
