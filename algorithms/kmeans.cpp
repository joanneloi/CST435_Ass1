#include "kmeans.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
void usage() {
    std::cout << "Usage: kmeans_METHOD --data taxi_100000.kmbin --centers initial.csv "
                 "[--k 5] [--dims 4] [--threads 4] [--max-iter 100] "
                 "[--tol 0.0001] [--output-prefix results/taxi] [--sample 3000]\n";
}
std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + argv[i]);
    return argv[++i];
}
void create_parent(const std::string& path) {
    const auto p = std::filesystem::path(path).parent_path();
    if (!p.empty()) std::filesystem::create_directories(p);
}
} // namespace

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; ++i) {
        const std::string a=argv[i];
        if (a=="--help" || a=="-h") { usage(); std::exit(0); }
        else if (a=="--data") o.data=next_arg(i,argc,argv);
        else if (a=="--centers") o.centers_file=next_arg(i,argc,argv);
        else if (a=="--output-prefix") o.output_prefix=next_arg(i,argc,argv);
        else if (a=="--k") o.k=std::stoi(next_arg(i,argc,argv));
        else if (a=="--dims") o.d=std::stoi(next_arg(i,argc,argv));
        else if (a=="--threads") o.threads=std::stoi(next_arg(i,argc,argv));
        else if (a=="--max-iter") o.max_iter=std::stoi(next_arg(i,argc,argv));
        else if (a=="--tol") o.tolerance=std::stod(next_arg(i,argc,argv));
        else if (a=="--sample") o.sample=std::stoi(next_arg(i,argc,argv));
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (o.data.empty() || o.centers_file.empty()) {
        usage(); throw std::runtime_error("--data and --centers are required");
    }
    if (o.k<1 || o.threads<1 || o.d<0 || o.max_iter<1 || o.sample<0 ||
        !std::isfinite(o.tolerance) || o.tolerance<0)
        throw std::runtime_error("Invalid K, dimensions, threads, iterations, sample or tolerance");
    return o;
}

Dataset read_kmbin(const std::string& path, int dimensions) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open dataset: " + path);
    char magic[8]; std::uint64_t n=0; std::uint32_t d=0;
    f.read(magic,8); f.read(reinterpret_cast<char*>(&n),8); f.read(reinterpret_cast<char*>(&d),4);
    if (!f || std::string(magic,8)!="KMBIN001" || n==0 || d==0 || d>16 || n>200000000ULL)
        throw std::runtime_error("Invalid dataset header (requires KMBIN001 and at most 16 features)");
    const int selected = dimensions==0 ? static_cast<int>(d) : dimensions;
    if (selected<1 || selected>static_cast<int>(d))
        throw std::runtime_error("--dims exceeds dataset feature count");
    Dataset data; data.n=n; data.d=selected;
    data.x.resize(static_cast<std::size_t>(n)*selected);
    if (selected==static_cast<int>(d)) {
        f.read(reinterpret_cast<char*>(data.x.data()),
               static_cast<std::streamsize>(data.x.size()*sizeof(double)));
        if (!f) throw std::runtime_error("Incomplete dataset data: " + path);
    } else {
        const std::size_t batch=8192;
        std::vector<double> buffer(batch*d);
        std::uint64_t done=0;
        while (done<n) {
            const std::size_t rows=static_cast<std::size_t>(std::min<std::uint64_t>(n-done,batch));
            f.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(rows*d*sizeof(double)));
            if (!f) throw std::runtime_error("Incomplete dataset data: " + path);
            for (std::size_t r=0;r<rows;++r)
                std::copy_n(buffer.data()+r*d,selected,data.x.data()+(done+r)*selected);
            done+=rows;
        }
    }
    for (double x:data.x) if (!std::isfinite(x)) throw std::runtime_error("Non-finite feature in dataset");
    return data;
}

std::vector<double> read_centers(const std::string& path,int k,int d) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open initial-centroid file: "+path);
    std::vector<double> c;
    std::string row;
    while (std::getline(f,row)) {
        if (row.empty()) continue;
        for (char& x:row) if (x==',') x=' ';
        std::istringstream stream(row);
        double value;
        while (stream >> value) {
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite initial centroid");
            c.push_back(value);
        }
        if (!stream.eof()) throw std::runtime_error("Invalid initial centroid CSV: " + path);
    }
    if (c.size()!=static_cast<std::size_t>(k)*d) {
        throw std::runtime_error("Initial centroid CSV must contain exactly K*D numbers (no header)");
    }
    return c;
}

int nearest_center(const double* x,const double* centers,int k,int d) {
    int closest=0; double best=std::numeric_limits<double>::infinity();
    for (int c=0;c<k;++c) {
        const double* center=centers+static_cast<std::size_t>(c)*d;
        double sum=0.0;
        for (int j=0;j<d;++j) { double dx=x[j]-center[j];sum+=dx*dx; }
        if (sum<best) { best=sum;closest=c; }
    }
    return closest;
}
double squared_distance(const double* a,const double* b,int d) {
    double sum=0;for (int j=0;j<d;++j) { const double z=a[j]-b[j];sum+=z*z; }return sum;
}
bool update_centers(std::vector<double>& centers,const std::vector<double>& sums,
                    const std::vector<std::uint64_t>& counts,int k,int d,double tol) {
    double max_shift_sq=0;
    for (int c=0;c<k;++c) {
        if (counts[c]==0) continue; // keep empty cluster unchanged, equally for every backend
        double shift=0;
        for (int j=0;j<d;++j) {
            const std::size_t idx=static_cast<std::size_t>(c)*d+j;
            const double next=sums[idx]/static_cast<double>(counts[c]);
            const double delta=centers[idx]-next;
            shift+=delta*delta; centers[idx]=next;
        }
        max_shift_sq=std::max(max_shift_sq,shift);
    }
    return max_shift_sq<=tol*tol;
}

void save_outputs(const Options& opt,const Dataset& data,const RunResult& r,
                  const char* method,double wcss,double total_ms) {
    if (opt.output_prefix.empty()) return;
    create_parent(opt.output_prefix+".centroids.csv");
    {
        std::ofstream f(opt.output_prefix+".centroids.csv");
        f<<"cluster";
        for (int d=0;d<data.d;++d) f<<",z_feature_"<<d;
        f<<'\n'<<std::setprecision(17);
        for (int k=0;k<opt.k;++k) {
            f<<k;
            for (int d=0;d<data.d;++d)f<<','<<r.centers[static_cast<std::size_t>(k)*data.d+d];
            f<<'\n';
        }
    }
    if (opt.sample>0) {
        std::ofstream f(opt.output_prefix+".sample.csv");
        for (int d=0;d<data.d;++d)f<<"z_feature_"<<d<<',';
        f<<"cluster\n"<<std::setprecision(12);
        const std::uint64_t count=std::min<std::uint64_t>(data.n,opt.sample);
        for (std::uint64_t i=0;i<count;++i) {
            const auto ix=(i*data.n)/count;
            const double* p=data.x.data()+ix*data.d;
            for (int d=0;d<data.d;++d) f<<p[d]<<',';
            f<<nearest_center(p,r.centers.data(),opt.k,data.d)<<'\n';
        }
    }
    std::ofstream f(opt.output_prefix+".summary.json");
    f<<std::setprecision(15)<<"{\n"
     <<"  \"method\": \""<<method<<"\",\n"
     <<"  \"N\": "<<data.n<<", \"K\": "<<opt.k<<", \"D\": "<<data.d
     <<", \"threads\": "<<((std::string(method)=="sequential")?1:opt.threads)<<",\n"
     <<"  \"iterations\": "<<r.iterations<<", \"converged\": "<<(r.converged?"true":"false")<<",\n"
     <<"  \"iteration_ms\": "<<r.iteration_ms<<", \"transfer_ms\": "<<r.transfer_ms
     <<", \"setup_ms\": "<<r.setup_ms<<", \"total_ms\": "<<total_ms
     <<", \"wcss\": "<<wcss<<"\n}";
}

int main(int argc,char** argv) {
    try {
        const Options opt=parse_options(argc,argv);
        const auto data=read_kmbin(opt.data,opt.d);
        if (static_cast<std::uint64_t>(opt.k)>data.n) throw std::runtime_error("K exceeds N");
        const auto initial=read_centers(opt.centers_file,opt.k,data.d);
        const auto start=Clock::now();
        const auto result=run_backend(data,opt,initial);
        if (result.centers.size()!=static_cast<std::size_t>(opt.k)*data.d)
            throw std::runtime_error("Implementation returned invalid centroid count");
        // Every method uses exactly the same final scoring pass on the CPU.
        // It is included in total_ms but not iterative-loop iteration_ms.
        double wcss=0.0;
        for (std::uint64_t i=0;i<data.n;++i) {
            const double* x=data.x.data()+i*data.d;
            const int c=nearest_center(x,result.centers.data(),opt.k,data.d);
            wcss+=squared_distance(x,result.centers.data()+static_cast<std::size_t>(c)*data.d,data.d);
        }
        const double total=ms(start,Clock::now());
        save_outputs(opt,data,result,backend_name(),wcss,total);
        std::cout<<std::setprecision(15)
          <<"RESULT_JSON {\"method\":\""<<backend_name()<<"\",\"N\":"<<data.n
          <<",\"K\":"<<opt.k<<",\"D\":"<<data.d
          <<",\"threads\":"<<((std::string(backend_name())=="sequential")?1:opt.threads)
          <<",\"iterations\":"<<result.iterations
          <<",\"converged\":"<<(result.converged?"true":"false")
          <<",\"iteration_ms\":"<<result.iteration_ms
          <<",\"transfer_ms\":"<<result.transfer_ms
          <<",\"setup_ms\":"<<result.setup_ms
          <<",\"total_ms\":"<<total
          <<",\"wcss\":"<<wcss<<"}"<<std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;
    }
}
