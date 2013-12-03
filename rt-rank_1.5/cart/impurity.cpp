#include <algorithm>
using namespace std;
 
#include "regression_tree.h"
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
using namespace boost;

extern vector<vector<int> > fInds;

int feature_to_split_on;

/*bool mysortpred(const tuple* d1, const tuple* d2) {
  return d1->features[feature_to_split_on] < d2->features[feature_to_split_on];
}
*/
/*void sort_data_by_feature(vector<tuple*>& data, int f) {
  feature_to_split_on = f;
  sort(data.begin(), data.end(), mysortpred);
}
*/

void sort_data_by_feature(vector<int>& location,  vector<int> dataCount, vector<int> invertIdx, int f){
	int cur=0;
        for (int i = 0; i < fInds[f].size(); i ++){
                int z = fInds[f][i];
		int loc=invertIdx[z];
                for (int j = 0; j < dataCount[z]; j ++)
			location[cur++]=loc;
        }
}


/*bool mysortpred2(const pair<tuple*, int> tk1, const pair<tuple*, int> tk2) {
  return tk1.first->features[tk1.second] < tk2.first->features[tk2.second];
}*/
/////////



//double best_fc_in_feature(vector<tuple*> data, int f, pair<int, double>& fc) {
double best_fc_in_feature(vector<tuple*> data, vector<int> dataCount, vector<int> invertIdx, int f, int& fs, double& vs, const args_t& args) {  //if (!XX)
/*  vector< pair<tuple*, int> > tk;
  int z;

  for (z = 0; z < data.size(); z++)
    tk.push_back( pair<tuple*,int>(data[z], f) );
  sort(tk.begin(), tk.end(), mysortpred2);
  for (z = 0; z < data.size(); z++)
    data[z] = tk[z].first;
*/
  int n = data.size(), i;
  double min = MY_DBL_MAX;
        vector<int> location(n, -1);
        sort_data_by_feature(location, dataCount, invertIdx, f);


  // get impurity (squared loss) for missing data
  double M = 0.0;
  double W = 0.0;
  int missing = 0;
	int loc = location[missing];
  while (data[loc]->features[f] == UNKNOWN && missing < n-1) {
    //M += data[missing]->target * 1.0;
	M += data[loc]->target * data[loc]->weight;
	W += data[loc]->weight;
    	missing++;
	loc = location[missing];
  }
  if (missing == n-1) // all data is missing
    return MY_DBL_MAX;
  if (missing) {
    //double mbar = M * 1.0 / missing;
	double mbar = M/W;
    M = 0.0;
    for (i = 0; i < missing; i++){
	loc = location[i];
      M += data[loc]->weight * (data[loc]->target - mbar) * (data[loc]->target - mbar);
    }
  }
  int nn = n - missing; // number of data points that arent missing
 
  // we have impurity = E_{i=0..n} (yi - ybar)^2  (E = summation)
  // this factors to impurty = E(yi^2) + (n+1)ybar^2 - 2*(n+1)*ybar^2
  // we want the impurity for left and right splits for all splits k
  // let ybl = ybar left, ybr = ybar right
  //     s = E_{i=0..k} yi^2
  //     r = E_{i=k+1..n} yi^2

  //impurity =\sum_i  wi (yi - ybar)^2 = \sum_i wi*yi^2 - 2*(\sum_i wi*yi)*ybar + (\sum_i wi) ybar^2 
  // ybar = (\sum_i wi*yi) / (sum_i wi) = ywr/WR
  //  r = \sum_i wi*yi^2, ywr = \sum_i wi*yi,  WR = \sum_i wi
  // impurity = r - 2*ywr*ywr/WR + ywr*ywr/WR = r - ywr*ywr/WR
  double ybl, ybr, s, r, L, R, I, ywl, ywr, WL, WR;
  ybl = ybr = s = r = ywl = ywr = WL = WR = 0.0;
  
  // put all data into right side R
  // get ybar right and r
  int start = missing;
  for (i = start; i < n; i++) {
	loc = location[i];
    	r += data[loc]->target * data[loc]->target * data[loc]->weight;
    	//ybr += data[i]->target;
    	ywr += data[loc]->target * data[loc]->weight;
    	WR += data[loc]->weight;
  }
  //ybr /= 1.0 * nn;
  //r += 0.0000000001 for precision errors
  
  // for every i
  // put yi into left side, remove it from the right side, and calculate squared lost
  // impurity of putting all data points into the right tree equals putting all the data into the left tree (such cases are not considered)
  for (i = start; i < n-1; i++) {
    int j = i - missing; 
    double yn = data[location[i]]->target;
    double w = data[location[i]]->weight;

    s += w *  yn * yn;
    r -= w * yn * yn;
    ywr -= w * yn;
    ywl += w * yn;
    WL += w;
    WR -= w;
    
    if (r < 0 && r > -0.000001) r = 0;
    if (r < 0) r = 0;
    
    //ybl = (j*ybl + yn) / (j+1.0);
    //ybr = ((nn-j)*ybr - yn) / (nn-j-1.0);

    //L = s + WL*ybl*ybl - 2*ybl*ywl;
    //R = r + WR*ybr*ybr - 2*ybr*ywr;
    L = s - ywl*ywl/WL;
    R = r - ywr*ywr/WR; 
    
    //L = s + (j+1)*ybl*ybl - 2*(j+1)*ybl*ybl;
    //R = r + (nn-j-1.0)*ybr*ybr - 2*(nn-j-1.0)*ybr*ybr;
    
    // precision errors?
    if (L < 0 && L > -0.0000001) L = 0; 
    if (R < 0 && R > -0.0000001) R = 0;
    if (R < 0) R = 0;
    if (L < 0) L = 0;

    if(0)
    if (L < 0 || R < 0 || r < 0)
      printf("Problem %lf %lf %lf\n", L, R, r);
    
    // do not consider splitting here if data is the same as next
    if (data[location[i]]->features[f] == data[location[i+1]]->features[f])
      continue;
    
    I = L + R;// + M;
    I = L + R + M;


    if (I < min) {
      min = I;
      fs = f;
      vs = (data[location[i]]->features[f] + data[location[i+1]]->features[f]) / 2;
    }
  }

  return min;
}


// find best feature to split on in range [start, end)
// store results int I, fs, vs : impurity, feature, value
//void find_split_in_range(vector<tuple*> data, vector<int> dataCount, vector<int> invertIdx, int start, int  end, int& fs, double& vs, double& I, vector<bool> skip, const args_t& args) {
void find_split_in_range(vector<tuple*> data, vector<int> dataCount, vector<int> invertIdx, pair<int,int> range, int& fs, double& vs, double& I, vector<bool>skip, const args_t& args) {

  double min = MY_DBL_MAX;
	int start = range.first, end = range.second;

  for (int i = start; i < end; i++) {
    int f = i + 1;
    if (skip[f]) continue;

    int fi;
    double vi, Ii;
    Ii = best_fc_in_feature(data, dataCount, invertIdx, f, fi, vi, args);
    if (Ii < min) {
      min = Ii;
      fs = fi;
      vs = vi;
    }
  }

  I = min;
}

bool find_split_p(vector<tuple*> data, vector<int> dataCount, vector<int> invertIdx, int NF, int& f_split, double& v_split, vector<bool>& skip, const args_t& args) {

  f_split = -1;
  double min = MY_DBL_MAX;
  int n = data.size(), i;

  pair<int, double>* fc = new pair<int,double>[NF];
  double* I = new double[NF];

  int numthreads = args.processors;
  thread** threads = new thread*[numthreads];
  
  int* F = new int[numthreads];
  double* V = new double[numthreads];
  double* Imp = new double[numthreads];

  for (i = 0; i < numthreads; i++)
//	find_split_in_range(data, dataCount, invertIdx, i*(NF-1)/numthreads, (i+1)*(NF-1)/numthreads, F[i], V[i], Imp[i], skip, args);
//    threads[i] = new thread(find_split_in_range, data, dataCount, invertIdx, i*(NF-1)/numthreads, (i+1)*(NF-1)/numthreads, ref(F[i]), ref(V[i]), ref(Imp[i]), ref(skip), cref(args));
	threads[i] = new thread(bind(find_split_in_range, data, dataCount, invertIdx, make_pair(i*(NF-1)/numthreads, (i+1)*(NF-1)/numthreads), ref(F[i]), ref(V[i]), ref(Imp[i]), ref(skip), cref(args) ));
  for (i = 0; i < numthreads; i++) {
    threads[i]->join(); 
    delete threads[i];
  }
  delete[] threads;

  for (i = 0; i < numthreads;i++)
    if (Imp[i] < min) {
      min = Imp[i];
      f_split = F[i];
      v_split = V[i];
    }

  delete[] fc;  delete[] I;  delete[] V;  delete[] Imp; delete[] F;
  return min != MY_DBL_MAX;
}

///////////////////////

bool dt_node::entropy_split(data_t data, vector<int> dataCount, vector<int> invertIdx, int NF, int& f_split, double& v_split, int K, bool par) {
  f_split = -1;
  double min = MY_DBL_MAX;
  int n = data.size(), i;
  
  vector<bool> skip;

  //min E(i=1..k) pi * log(1/pi)
  
  for (i = 0; i <= NF; i++)
    skip.push_back( (K > 0) ? true : false);

  for (i = 0; i < K; i++) {
    int f;
    do
      f = rand() % (NF-2) + 1;
    while (!skip[f]);
    skip[f] = false;
  }


  //if (K <= 0)
  //return find_split_p(data, NF, f_split, v_split, skip);

  vector<int> location(n, -1);
  for (int f = 1; f < NF; f++) {
	if (skip[f]) continue;
	sort_data_by_feature(location, dataCount, invertIdx, f);
    //sort_data_by_feature(data,f);

    //if (skip[f]) continue;

    /*
 vector< pair<tuple*, int> > tk;
  int z;

  for (z = 0; z < data.size(); z++)
  tk.push_back( pair<tuple*,int>(data[z], f) );
  sort(tk.begin(), tk.end(), mysortpred2);


  for (z = 0; z < data.size(); z++)
  data[z] = tk[z].first;
    */


  int num_c = 7;
  vector<double> c_miss, c_left, c_right;
  //vector<double> p_miss, p_left, p_right;
  //vector<int> freq;
  //int n_left = 0, n_right = 0;
  for (i=0;i<num_c;i++) {
	c_miss.push_back(0.0);
//	p_miss.push_back(0.0);
    c_left.push_back(0.0);
    c_right.push_back(0.0);
//    p_left.push_back(0.0);
//    p_right.push_back(0.0);
//    freq.push_back(0);
  }



    // get impurity (entropy) for missing data
    double M = 0.0, L, R, W_miss = 0.0, W_left = 0.0, W_right = 0.0;
    int missing = 0;
	int loc = location[missing];
    while (data[loc]->features[f] == UNKNOWN && missing < n-1) {
	c_miss[(int)data[loc]->target] += data[loc]->weight;
	W_miss += data[loc]->weight;
        missing++;
	loc = location[missing];
    }

    if (missing == n-1) // all data is missing
      continue;

    int nn = n - missing; // number of data points that arent missing
    // entropy
    // put all data into right side R
    int start = missing;
    for (i = start; i < n; i++) {
	loc = location[i];
      	c_right[(int)data[loc]->target]+=data[loc]->weight;
      	W_right+=data[loc]->weight;
    }
	if (W_right){
    		for (i = 0; i < num_c; i++) {
      			if (c_right[i])
				R += c_right[i]/W_right * log(W_right / c_right[i]);
    		}
	}

    	if (missing && W_miss) {
                for (i = 0; i < num_c; i ++){
                        if (c_miss[i])
                                M += c_miss[i]/W_miss * log(W_miss/c_miss[i]);
                }
    	}
    	L = 0.0;

    // for every i
    // put yi into left side, remove it from the right side, and calculate squared lost
    for (i = start; i < n-1; i++) {
      int j = i - missing; 
      int yn = (int)data[loc]->target;

	W_right -= data[loc]->weight;
	W_left += data[loc]->weight;
      //n_right--;
      //n_left++;
	c_left[yn] += data[loc]->weight;
	c_right[yn] -= data[loc]->weight;
      //c_left[yn]++;
      //c_right[yn]--;

      //p_left[yn] += 1.0 / freq[yn];
      //p_right[yn] -= 1.0 / freq[yn];

      // do not consider splitting here if data is the same as next
      if (data[location[i]]->features[f] == data[location[i+1]]->features[f])
	continue;

      L = 0.0;
      int k;
      for (k = 0; k < num_c; k++) {
      	if (c_left[k])
	  L +=  c_left[k]/W_left * log(W_left / c_left[k]); 
      }
      R = 0.0;
      for (k = 0; k < num_c; k++) {
	if (c_right[k])
	  R += c_right[k]/W_right * log(W_right / c_right[k]);   
      }
      double ssum = W_left+W_right+W_miss;
      double I = W_left/ssum * L +  W_right/ssum * R + W_miss/ssum * M;
      //I = 1.0*n_left/n * 

      /*
      L = 0.0, R = 0.0;
      for (i = 0; i < num_c; i++)
	L += p_left;
      */

      if (I < min) {
	min = I;
	f_split = f;
	v_split = (data[location[i]]->features[f] + data[location[i+1]]->features[f])/2;
      }
    }
  }

  return min != MY_DBL_MAX;
}






//////////////////////


bool dt_node::find_split(vector<tuple*> data, vector<int> dataCount, vector<int> invertIdx, int NF, int& f_split, double& v_split, int K, bool par, const args_t& args) {
  if (args.loss == ALG_ENTROPY)
    return entropy_split(data, dataCount, invertIdx, NF, f_split, v_split, K, par);

  f_split = -1;
  double min = MY_DBL_MAX;
  int n = data.size(), i;
  
  vector<bool> skip;

  //K = NF/2;

  // pick K random features to split on, if specified
  for (i = 0; i <= NF; i++)
    skip.push_back( (K > 0) ? true : false);
  for (i = 0; i < K; i++) {
    int f;
    do
      f = rand() % (NF-2) + 1;
    while (!skip[f]);
    skip[f] = false;
  }

  //if (K >= 10)
  if (args.alg != ALG_FOREST && args.processors!=1)
    return find_split_p(data, dataCount, invertIdx, NF, f_split, v_split, skip, args);

	vector<int> location(n, -1);
  for (int f = 1; f < NF; f++) {
        if (skip[f]) continue;
    // sort data
/*    vector< pair<tuple*, int> > tk;
    int z;
    for (z = 0; z < data.size(); z++)
      tk.push_back( pair<tuple*,int>(data[z], f) );
    sort(tk.begin(), tk.end(), mysortpred2);
    for (z = 0; z < data.size(); z++)
      data[z] = tk[z].first;
*/
	sort_data_by_feature(location, dataCount, invertIdx, f);

    // get impurity (squared loss) for missing data
/*    double M = 0.0;
    int missing = 0;
    while (data[missing]->features[f] == UNKNOWN && missing < n-1) {
      M += data[missing]->target * 1.0;
      missing++;
    }
    if (missing == n-1) // all data is missing
      continue;
    if (missing) {
      double mbar = M * 1.0 / missing;
      M = 0.0;
      for (i = 0; i < missing; i++)
	M += (data[i]->target - mbar) * (data[i]->target - mbar);
    }
    int nn = n - missing; // number of data points that arent missing
*/
  double M = 0.0;
  double W = 0.0;
  int missing = 0;
	int loc = location[missing];
  while (data[loc]->features[f] == UNKNOWN && missing < n-1) {
    //M += data[missing]->target * 1.0;
        M += data[loc]->target * data[loc]->weight;
        W += data[loc]->weight;
    	missing++;
	loc = location[missing];
  }
  if (missing == n-1) // all data is missing
    return MY_DBL_MAX;
  if (missing) {
    //double mbar = M * 1.0 / missing;
        double mbar = M/W;
    M = 0.0;
    for (i = 0; i < missing; i++){
	loc = location[i];
      M += data[loc]->weight * (data[loc]->target - mbar) * (data[loc]->target - mbar);
   }
  }
 
  double ybl, ybr, s, r, L, R, I, ywl, ywr, WL, WR;
  ybl = ybr = s = r = ywl = ywr = WL = WR = 0.0;

  // put all data into right side R
  // get ybar right and r
  int start = missing;
  for (i = start; i < n; i++) {
	loc = location[i];
    	r += data[loc]->target * data[loc]->target * data[loc]->weight;
    	//ybr += data[i]->target;
    	ywr += data[loc]->target * data[loc]->weight;
    	WR += data[loc]->weight;
  }
  //ybr /= 1.0 * nn;
  //r += 0.0000000001 for precision errors

  // for every i
  // put yi into left side, remove it from the right side, and calculate squared lost
  // impurity of putting all data points into the right tree equals putting all the data into the left tree (such cases are not considered)
  for (i = start; i < n-1; i++) {
    int j = i - missing;
    double yn = data[location[i]]->target;
    double w = data[location[i]]->weight;

    s += w *  yn * yn;
    r -= w * yn * yn;
    ywr -= w * yn;
    ywl += w * yn;
    WL += w;
    WR -= w;

    if (r < 0 && r > -0.000001) r = 0;
    if (r < 0) r = 0;
    //ybl = (j*ybl + yn) / (j+1.0);
    //ybr = ((nn-j)*ybr - yn) / (nn-j-1.0);
    //L = s + WL*ybl*ybl - 2*ybl*ywl;
    //R = r + WR*ybr*ybr - 2*ybr*ywr;
    L = s - ywl*ywl/WL;
    R = r - ywr*ywr/WR;
    // precision errors?
    if (L < 0 && L > -0.0000001) L = 0;
    if (R < 0 && R > -0.0000001) R = 0;
    if (R < 0) R = 0;
    if (L < 0) L = 0;

    if(0)
    if (L < 0 || R < 0 || r < 0)
      printf("Problem %lf %lf %lf\n", L, R, r);

    // do not consider splitting here if data is the same as next
    if (data[location[i]]->features[f] == data[location[i+1]]->features[f])
      continue;

    I = L + R + M ;

      if (I < min) {
        min = I;
        f_split = f;
        v_split = (data[location[i]]->features[f] + data[location[i+1]]->features[f])/2;
      }

  }
}
  return min != MY_DBL_MAX;
}

