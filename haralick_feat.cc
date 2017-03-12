#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>

#define EPS 0.000001

using namespace cv;
using namespace std;


//Entropy of vector<double>
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


//Marginal probabilities as in px = sum on j(p(i, j))
//                             py = sum on i(p(i, j))
vector<double> MargProbx(Mat cooc) {
    vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[i] += cooc.at<double>(i, j);
    return result;
}

vector<double> MargProby(Mat cooc) {
    vector<double> result(cooc.cols, 0.0);
    for (int j = 0; j < cooc.cols; j++)
        for (int i = 0; i < cooc.rows; i++)
            result[j] += cooc.at<double>(i, j);
    return result;
}

//probsum  := Px+y(k) = sum(p(i,j)) given that i + j = k
vector<double> ProbSum(Mat cooc) {
    vector<double> result(cooc.rows * 2, 0.0);
    for (int i = 0; i < cooc.rows; i++) 
        for (int j = 0; j < cooc.cols; j++)
            result[i + j] += cooc.at<double>(i, j);
    return result;
}

//probdiff := Px-y(k) = sum(p(i,j)) given that |i - j| = k
vector<double> ProbDiff(Mat cooc) {
    vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[abs(i - j)] += cooc.at<double>(i, j);
    return result;
}


/*Features from coocurrence matrix*/
double HaralickEnergy(Mat cooc) {
    double energy = 0;
    for (int i = 0; i < cooc.rows; i++) {
        for (int j = 0; j < cooc.cols; j++) {
            energy += cooc.at<double>(i,j) * cooc.at<double>(i,j);
        }
    }
    return energy;
}

double HaralickEntropy(Mat cooc) {
    double entrop = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            entrop += cooc.at<double>(i,j) * log(cooc.at<double>(i,j) + EPS);
    return -1 * entrop;
}

double HaralickInverseDifference(Mat cooc) {
    double res = 0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            res += cooc.at<double>(i, j) * (1 / (1 + (i - j) * (i - j)));
    return res;
}

/*Features from MargProbs */
double HaralickCorrelation(Mat cooc, vector<double> probx, vector<double> proby) {
    double corr;
    double meanx, meany, stddevx, stddevy;
    meanStd(probx, meanx, stddevx);
    meanStd(proby, meany, stddevy);
    for (int i = 0; i < cooc.rows; i++) 
        for (int j = 0; j < cooc.cols; j++)
            corr += (i * j * cooc.at<double>(i, j)) - meanx * meany;
    return corr / (stddevx * stddevy);
}

//InfoMeasure1 = HaralickEntropy - HXY1 / max(HX, HY)
//HXY1 = sum(sum(p(i, j) * log(px(i) * py(j))
double HaralickInfoMeasure1(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
    double hx = Entropy(probx);
    double hy = Entropy(proby);
    double hxy1 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy1 += cooc.at<double>(i, j) * log(probx[i] * proby[j] + EPS);
    hxy1 = -1 * hxy1;

    return (ent - hxy1) / max(hx, hy);

}

//InfoMeasure2 = sqrt(1 - exp(-2(HXY2 - HaralickEntropy)))
//HX2 = sum(sum(px(i) * py(j) * log(px(i) * py(j))
double HaralickInfoMeasure2(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
    double hxy2 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy2 += probx[i] * proby[j] * log(probx[i] * proby[j] + EPS);
    hxy2 = -1 * hxy2;

    return sqrt(1 - exp(-2 * (hxy2 - ent)));
}

/*Features from ProbDiff*/
double HaralickContrast(Mat cooc, vector<double> diff) {
    double contrast = 0.0;
    for (int i = 0; i < diff.size(); i++) 
        contrast += i * i * diff[i];
    return contrast;
}

double HaralickDiffEntropy(Mat cooc, vector<double> diff) {
    double diffent = 0.0;
    for (int i = 0; i < diff.size(); i++) 
        diffent += diff[i] + log(diff[i] + EPS);
    return -1 * diffent;
}

double HaralickDiffVariance(Mat cooc, vector<double> diff) {
    double diffvar = 0.0;
    double diffent = HaralickDiffEntropy(cooc, diff);
    for (int i = 0; i < diff.size(); i++)
        diffvar += (i - diffent) * (i - diffent) * diff[i];
}

/*Features from Probsum*/
double HaralickSumAverage(Mat cooc, vector<double> sumprob) {
    double sumav = 0.0;
    for (int i = 0; i < sumprob.size(); i++)
        sumav += i * sumprob[i];
    return sumav;
}

double HaralickSumEntropy(Mat cooc, vector<double> sumprob) {
    double sument = 0.0;
    for (int i = 0; i < sumprob.size(); i++) 
        sument += sumprob[i] * log(sumprob[i] + EPS);
    return -1 * sument;
}

double HaralickSumVariance(Mat cooc, vector<double> sumprob) {
    double sumvar = 0.0;
    double sument = HaralickSumEntropy(cooc, sumprob);
    for (int i = 0; i < sumprob.size(); i++)
        sumvar += (i - sument) * (i - sument) * sumprob[i];
    return sumvar;
}


Mat MatCooc(Mat img, int N, int deltax, int deltay) 
{
    int atual, vizinho;
    int newi, newj;
    Mat ans = Mat::zeros(N + 1, N + 1, CV_64F);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            newi = i + deltay;
            newj = j + deltax;
            if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
                atual = (int) img.at<uchar>(i, j);
                vizinho = (int) img.at<uchar>(newi, newj);
                ans.at<double>(atual, vizinho) += 1.0;
            }
        }
    }
    return ans / (img.rows * img.cols);
}

//Assume tamanho deltax == tamanho deltay 
Mat MatCoocAdd(Mat img, int N, std::vector<int> deltax, std::vector<int> deltay)
{
    Mat ans, nextans;
    ans = MatCooc(img, N, deltax[0], deltay[0]);
    for (int i = 1; i < deltax.size(); i++) {
        nextans = MatCooc(img, N, deltax[i], deltay[i]);
        add(ans, nextans, ans);
    }
    return ans;
}

void printMat(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++)
            printf("%lf ", (double) img.at<double>(i, j));
        printf("\n");
    }
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Insira uma imagem\n");
        return 0;
    }
    int N;
    double min, max;
    std::vector<int> deltax({1});
    std::vector<int> deltay({0});
    Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat ans = MatCoocAdd(img, 255, deltax, deltay);
    std::vector<double> sum = ProbSum(ans);
    std::vector<double> diff = ProbDiff(ans);
    std::vector<double> probx = MargProbx(ans);
    std::vector<double> proby = MargProby(ans);
    double ent = HaralickEntropy(ans);
    cout << "Energy: " << HaralickEnergy(ans) << endl;
    cout << "Entropy: " << ent << endl;
    cout << "Inverse Difference Moment: " << HaralickInverseDifference(ans) << endl;
    cout << "Correlation: " << HaralickCorrelation(ans, probx, proby) << endl;
    cout << "Info Measure of Correlation 1: " << HaralickInfoMeasure1(ans, ent, probx, proby) << endl;
    cout << "Info Measure of Correlation 2: " << HaralickInfoMeasure2(ans, ent, probx, proby) << endl;
    cout << "Contrast: " << HaralickContrast(ans, diff) << endl;
    cout << "Difference Variance: " << HaralickDiffVariance(ans, diff) << endl;
    cout << "Difference Entropy: " << HaralickDiffEntropy(ans, diff) << endl;
    cout << "Sum Average: " << HaralickSumAverage(ans, sum) << endl;
    cout << "Sum Variance: " << HaralickSumVariance(ans, sum) << endl;
    cout << "Sum Entropy: " << HaralickSumEntropy(ans, sum) << endl;

    //minMaxLoc(ans, &min, &max);
    //ans = 255 * (ans/max);
    //ans.convertTo(ans, CV_8UC1);
    //imshow("cooc", ans);
    //waitKey(0);
    //printMat(ans);
    return 0;
} 
    
  
