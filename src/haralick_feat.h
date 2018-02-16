#include <opencv2/opencv.hpp>

#include <iterator>
#include <vector>
#include <cmath>
#include <math.h>

#define EPS 0.00000001

using namespace cv;
using namespace std;

bool OutofBounds(int i, int j, Mat img) {
    return (i > img.rows || i < 0 && j > img.cols && j < 0);
}

double Entropy(vector<double> vec) {
    double result = 0.0;
    for (int i = 0; i < vec.size(); i++)
        result += vec[i] * log(vec[i] + EPS);
    return -1 * result;
}

void meanStd(vector<double> v, double &m, double &stdev) {
    double sum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        sum += d;
    });
    m =  sum / v.size();

    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });

    stdev = sqrt(accum / (v.size()-1));
}

class HaralickExtractor {
    private:
        Mat matcooc; //GLCM
        vector<double> margprobx;
        vector<double> margproby;
        vector<double> probsum; //sum probability
        vector<double> probdiff; //diff probability
        double hx, hy; //entropy of margprobx and y
        double meanx, meany, stddevx, stddevy;
        bool initial=false; //marks if above variables are set
        
        /*calculates probsum, probdiff, margprobx and y at once*/
        void fast_init() {
            if (matcooc.empty())
                return;
            margprobx.clear();
            margprobx.resize(matcooc.rows, 0.0);
            margproby.clear();
            margproby.resize(matcooc.cols, 0.0);
            probsum.clear();
            probsum.resize(matcooc.rows * 2, 0.0);
            probdiff.clear();
            probdiff.resize(matcooc.rows, 0.0); 

            double local;
            for (int i = 0; i < matcooc.rows; i++) {
                for (int j = 0; j < matcooc.cols; j++) {
                    local = matcooc.at<double>(i, j);
                    margprobx[i] += local;
                    margproby[j] += local;
                    probsum[i + j] += local;
                    probdiff[abs(i - j)] += local;
                }
            }
            hx = Entropy(margprobx);
            hy = Entropy(margproby);
            meanStd(margprobx, meanx, stddevx);
            meanStd(margproby, meany, stddevy);
            //Everything set up
            initial = true;
        }
  
        /*0 => energy, 1 => entropy, 2=> inverse difference */
        /*3 => correlation, 4=> info measure 1, 5 => info measure 2*/
        vector<double> cooc_feats() {
            vector<double> ans(6, 0.0);
            double hxy1 = 0.0;
            double hxy2 = 0.0;
            double local;
            for (int i = 0; i < matcooc.rows; i++) {
                for (int j = 0; j < matcooc.cols; j++) {
                    local = matcooc.at<double>(i, j);
                    ans[0] += local * local;
                    ans[1] += local * log(local + EPS);
                    ans[2] += local * (1 / (1 + (i - j) * (i - j)));
                    ans[3] += (i * j * local) - (meanx * meany);
                    hxy1 += local * log(margprobx[i] * margproby[j] + EPS);
                    hxy2 += margprobx[i] * margproby[j] * log(margprobx[i] * margproby[j] + EPS);
                }
            }
            hxy1 = hxy1 * -1;
            hxy2 = hxy2 * -1;
            ans[1] = -1 * ans[1];
            ans[3] = ans[3] / (stddevx * stddevy);
            ans[4] = (ans[1] - hxy1) / max(hx, hy);
            ans[5] = sqrt(1 - exp(-2 *(hxy2 - ans[1])));
            return ans;
        }

        /*0 => contrast, 1 => diff entropy, 2 => diffvariance */
        /*3 => sum average, 4 => sum entropy, 5 => sum variance */
        vector<double> margprobs_feats() {
            vector<double> ans(6, 0.0);
            for (int i = 0; i < probdiff.size(); i++) {
                ans[0] += i * i * probdiff[i];
                ans[1] += -1 * probdiff[i] * log(probdiff[i] + EPS);
            }
            for (int i = 0; i < probsum.size(); i++) {
                ans[3] += i * probsum[i];
                ans[4] += -1 * probsum[i] * log(probsum[i] + EPS);
            }
            for (int i = 0; i < probdiff.size(); i++) 
                ans[2] += (i - ans[1]) * (i - ans[1]) * probdiff[i];
            for (int i = 0; i < probsum.size(); i++)
                ans[5] += (i - ans[4]) * (i - ans[4]) * probsum[i];
            return ans;               
        }


    
    public:
        vector<double> fast_feats(bool verbose=false) {
            vector<double> result(12, 0.0);          
            if (matcooc.empty()) {
                return result;
            }
            if (!initial)
                fast_init();
            vector<double> margfeats = margprobs_feats();
            vector<double> coocfeats = cooc_feats();
            for (int i = 0; i < 6; i++)
                result[i] = coocfeats[i];
            for (int i = 0; i < 6; i++)
                result[6 + i] = margfeats[i];
            return result;
        }
        
        Mat MatCooc(Mat img, int N, int deltax, int deltay) {
            int target, next;
            int newi, newj;
            Mat ans = Mat::zeros(N + 1, N + 1, CV_64F);
            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    newi = i + deltay;
                    newj = j + deltax;
                    if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0){
                        target = (int) img.at<uchar>(i, j);
                        next = (int) img.at<uchar>(newi, newj);
                        ans.at<double>(target, next) += 1.0;
                    }
                }
            }
            return ans / (img.rows * img.cols);
        }

        Mat MatCoocAdd(Mat img, int N, vector<int> deltax, vector<int> deltay) {
            Mat ans, nextans;
            ans = MatCooc(img, N, deltax[0], deltay[0]);
            for (int i = 1; i < deltax.size(); i++) {
                nextans = MatCooc(img, N, deltax[i], deltay[i]);
                add(ans, nextans, ans);
            }
            return ans;
        }

        vector<double> getFeaturesFromImage(Mat img, vector<int> deltax, vector<int> deltay, bool verbose=false) {
            if (img.type() != CV_8UC1) {
                cout << "Unsupported image type" << endl;
                return vector<double>(0);
            }
            matcooc = MatCoocAdd(img, 255, deltax, deltay);
            fast_init(); //initialize internal variables
            vector<double> ans = fast_feats();
            if (verbose) {
                cout << "Energy: " << ans[0] << endl;
                cout << "Entropy: " << ans[1] << endl;
                cout << "Inverse Difference Moment: " << ans[2] << endl;
                cout << "Correlation: " << ans[3] << endl;
                cout << "Info Measure of Correlation 1: " << ans[4] << endl;
                cout << "Info Measure of Correlation 2:" << ans[5] << endl;
                cout << "Contrast: " << ans[6] << endl;
                cout << "Difference Entropy: " << ans[7] << endl;
                cout << "Difference Variance: " << ans[8] << endl;
                cout << "Sum Average: " << ans[9] << endl;
                cout << "Sum Entropy: " << ans[10] << endl;
                cout << "Sum Variance: " << ans[11] << endl;  
            }
            return ans;
        }

        //Constructor for use on single image
        //img is a grayscale image, deltax and deltay are pairs of the directions
        //to which we want to make the GLCM
        //temporarily accepting only CV_8UC1
        HaralickExtractor(Mat img, vector<int> deltax, vector<int> deltay) {
            if (img.type() != CV_8UC1) {
                cout << "Unsupported image type" << endl;
                return;
            }
            matcooc = MatCoocAdd(img, 255, deltax, deltay);
        }

        //Constructor for use on various images
        HaralickExtractor() {
            return;
        }
};
