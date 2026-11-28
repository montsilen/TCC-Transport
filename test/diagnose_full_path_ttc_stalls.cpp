// Replay the production controller and inspect the last repeated state.
#include "common_window_ttc_kernel.cpp"

namespace {
struct OfferCounts {
    int tested=0, safe_now=0, safe_later=0, moving=0, unilateral=0, exiting=0;
};

OfferCounts count_offers(const Path*paths,const Size*sizes,const State*state,bool continuous,bool add_stop) {
    Banks banks;
    for(int j=0;j<2;j++) build_bank(banks[j],paths[j],state[j],continuous,true,add_stop);
    OfferCounts counts;
    for(const auto& a:banks[0]) for(const auto& b:banks[1]) {
        ++counts.tested;
        if(!safe(a,b,paths,sizes,2,true)) continue;
        ++counts.safe_now;
        if(!safe(a,b,paths,sizes,60,true)) continue;
        ++counts.safe_later;
        const bool move_a=a.trajectory[replan].p>state[0].p+1e-7;
        const bool move_b=b.trajectory[replan].p>state[1].p+1e-7;
        counts.moving+=move_a||move_b;
        counts.unilateral+=move_a!=move_b;
        counts.exiting+=std::isfinite(a.arrival)||std::isfinite(b.arrival);
    }
    return counts;
}

std::array<int,2> finer_speed_probe(const Path*paths,const Size*sizes,const State*state) {
    Banks banks;
    for(int j=0;j<2;j++)
        for(double target:{0.,.25,.5,1.,2.,3.,4.,5.,6.,7.,10.,13.,15.})
            for(double ramp:{.25,.5,1.}) {
                if (target==0&&ramp!=1) continue;
                banks[j].push_back(profile(paths[j],state[j],{0,target,ramp},true));
            }
    std::array<int,2> result{0,0};
    for(const auto& a:banks[0]) for(const auto& b:banks[1]) {
        const bool moving_a=a.trajectory[replan].p>state[0].p+1e-7;
        const bool moving_b=b.trajectory[replan].p>state[1].p+1e-7;
        if (!(moving_a||moving_b)||!safe(a,b,paths,sizes,60,true)) continue;
        ++result[0];
        result[1]+=std::isfinite(a.arrival)||std::isfinite(b.arrival);
    }
    return result;
}

bool clear_if_other_parks(const Path* paths,const Size*sizes,const State*state,int moving) {
    const int other=1-moving;
    const Pose parked=paths[other].at(state[other].p);
    const double span=paths[moving].end()-state[moving].p;
    const int samples=std::max(1,int(std::ceil(span/.05)));
    for(int i=0;i<=samples;i++) {
        const Pose at=paths[moving].at(state[moving].p+span*i/samples);
        if (gap_axis(at,parked,sizes[moving],sizes[other])<=1e-6) return false;
    }
    return true;
}
}

extern "C" __declspec(dllexport) int diagnose_stall(const double *path_a,int count_a,const double *path_b,int count_b,
        const double *entry,const double *exit,const double *dimensions,const double *initial,
        int arm,double *metrics,double *trajectory,int capacity,double *terminal) {
    if (count_a<2||count_b<2||(arm!=0&&arm!=1)) return -1;
    Path paths[2]{{path_a,count_a,entry[0],exit[0]},{path_b,count_b,entry[1],exit[1]}};
    const Size sizes[2]{{dimensions[0]/2,dimensions[1]/2},{dimensions[2]/2,dimensions[3]/2}};
    const double goals[]{paths[0].end(),paths[1].end()};
    std::vector<Decision> traces;
    const Metrics m=simulate(paths,sizes,initial,0,arm,goals,true,&traces);
    serialize(metrics,m);
    metrics[13]=m.exposure;metrics[14]=m.elapsed;metrics[15]=m.stalled;
    if(!m.stalled||traces.empty()) return -2;
    if(int(traces.size())>capacity) return -3;
    for (std::size_t i=0;i<traces.size();i++) {
        const auto& row=traces[i];
        const auto& a=row.before[0];const auto& b=row.before[1];
        const auto& c=row.after[0];const auto& d=row.after[1];
        const auto& u=row.offers[0];const auto& v=row.offers[1];
        double record[]{row.tick*dt,row.steps*dt,double(row.fallback),double(row.feasible_pairs),
            a.p,a.v,b.p,b.v,c.p,c.v,d.p,d.v,
            row.fallback?inf:u.wait,row.fallback?inf:u.target,row.fallback?inf:u.ramp,
            row.fallback?inf:v.wait,row.fallback?inf:v.target,row.fallback?inf:v.ramp};
        std::copy(std::begin(record),std::end(record),trajectory+i*18);
    }
    const State* state=traces.back().after.data();
    const double start_gap=box_gap(paths[0].at(0),paths[1].at(0),sizes[0],sizes[1],inf);
    const double end_gap=box_gap(paths[0].at(state[0].p),paths[1].at(state[1].p),sizes[0],sizes[1],inf);
    const auto original=count_offers(paths,sizes,state,arm==0,false);
    const auto with_stop=arm==0?original:count_offers(paths,sizes,state,false,true);
    const auto finer=end_gap>1e-9?finer_speed_probe(paths,sizes,state):std::array<int,2>{0,0};
    const double record[]{entry[0],entry[1],goals[0],goals[1],start_gap,end_gap,ttc(paths,state,sizes),
        double(original.tested),double(original.safe_now),double(original.safe_later),double(original.moving),
        double(original.unilateral),double(original.exiting),
        double(with_stop.tested),double(with_stop.safe_now),double(with_stop.safe_later),double(with_stop.moving),
        double(with_stop.unilateral),double(with_stop.exiting),
        double(clear_if_other_parks(paths,sizes,state,0)),double(clear_if_other_parks(paths,sizes,state,1)),
        double(finer[0]),double(finer[1])};
    std::copy(std::begin(record),std::end(record),terminal);
    return int(traces.size());
}
