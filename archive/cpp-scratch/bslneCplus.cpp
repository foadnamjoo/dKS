
#include <iostream>

#include <vector>

#include <algorithm>  

#include <random>

#include <chrono>

#include <cmath>   

        

using namespace std;

        

struct Point {

    double x, y;

    int label;   // +1 for P, -1 for Q

};

             

double baselineKS(const vector<Point>& pts) {

    int n = pts.size();

    

    vector<Point> byY = pts;

    sort(byY.begin(), byY.end(), [](const Point& a, const Point& b){

        return a.y < b.y;

    });

    

    vector<double> xs;

    for (auto &p : pts) xs.push_back(p.x);

    

    double M = 0.0;

        

    for (double x0 : xs) {

        int s = 0;

        for (auto &p : byY) {

            if (p.x <= x0) {

                s += p.label;

                M = max(M, (double)abs(s));

            }

        }

    }

    return M / (n/2.0);  

}

        

vector<Point> makeUniform(int n) {

    mt19937 gen(123);

    uniform_real_distribution<> dist(0.0,1.0);

             



    vector<Point> pts;

    for(int i=0;i<n;i++){

        pts.push_back({dist(gen), dist(gen), +1});

        pts.push_back({dist(gen), dist(gen), -1});

    }

    return pts;   

}

            

vector<Point> makeGaussian(int n) {

    mt19937 gen(123);

    normal_distribution<> dist(0.0,1.0);

         

    vector<Point> pts;

    for(int i=0;i<n;i++){

        pts.push_back({dist(gen), dist(gen), +1});

        pts.push_back({dist(gen), dist(gen), -1});

    }

    return pts;

}

          



int main() {

    vector<int> ns = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

        

    for (int n : ns) {

        auto u = makeUniform(n);

        auto g = makeGaussian(n);

 

        auto t1 = chrono::high_resolution_clock::now();

        baselineKS(u);

        auto t2 = chrono::high_resolution_clock::now();

        double tu = chrono::duration<double>(t2-t1).count();

         

        t1 = chrono::high_resolution_clock::now();

        baselineKS(g);   

        t2 = chrono::high_resolution_clock::now();

        double tg = chrono::duration<double>(t2-t1).count();

     

        cout << "n=" << n

             << "  Uniform=" << tu

             << "  Gaussian=" << tg << endl;

    }

}


