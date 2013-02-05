#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <cv.h>
#include <cxcore.h>
#include <cmath>
#include <iostream>

using namespace std;


class kalman 
{
public:
	void init_kalman(int x, int xv, int y, int yv);
	CvKalman* cvkalman;
	CvMat* measurement;
	CvMat* prediction;
	CvPoint2D32f get_predict();
	void refine(float x, float y);
	void release();
	kalman(int x=0,int xv=0,int y=0,int yv=0);
	//virtual ~kalman();


};

CvRandState rng;
const double T = 1;
kalman::kalman(int x, int xv,int y, int yv)
{     
	cvkalman = cvCreateKalman( 4, 4, 0 );
    measurement = cvCreateMat( 4, 1, CV_32FC1 );

	/* create matrix data */
	const float A[] = {
		1, T, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, T,
		0, 0, 0, 1
	};

	const float H[] = {
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 0
	};

	const float P[] = {
		pow(4.0 ,2), pow(4.0,2)/T, 0, 0,
		pow(4.0 ,2)/T, pow(4.0,2)/pow(T,2), 0, 0,
		0, 0, pow(3.0,2), pow(3.0,2)/T,
		0, 0, pow(3.0,2)/T, pow(3.0,2)/pow(T,2)
	};

	const float Q[] = {
		pow(T,3)/3, pow(T,2)/2, 0, 0,
		pow(T,2)/2, T, 0, 0,
		0, 0, pow(T,3)/3, pow(T,2)/2,
		0, 0, pow(T,2)/2, T
	};

	const float R[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};


	cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );

	cvZero( measurement );

	cvRandSetRange( &rng, 0, 0.1, 0 );
	rng.disttype = CV_RAND_NORMAL;

	memcpy( cvkalman->transition_matrix->data.fl, A, sizeof(A));
	memcpy( cvkalman->measurement_matrix->data.fl, H, sizeof(H));
	memcpy( cvkalman->process_noise_cov->data.fl, Q, sizeof(Q));
	memcpy( cvkalman->error_cov_post->data.fl, P, sizeof(P));
	memcpy( cvkalman->measurement_noise_cov->data.fl, R, sizeof(R));
	//cvSetIdentity( cvkalman->process_noise_cov, cvRealScalar(1e-5) );   
	//cvSetIdentity( cvkalman->error_cov_post, cvRealScalar(1));
	//cvSetIdentity( cvkalman->measurement_noise_cov, cvRealScalar(1e-1) );

	/* choose initial state */

	cvkalman->state_post->data.fl[0]=x;
	cvkalman->state_post->data.fl[1]=xv;
	cvkalman->state_post->data.fl[2]=y;
	cvkalman->state_post->data.fl[3]=yv;

}

CvPoint2D32f kalman::get_predict()
{

	const CvMat* prediction = cvKalmanPredict(cvkalman, 0 );
	float predict_value_x = prediction->data.fl[0];
	float predict_value_y = prediction->data.fl[2];

	return(cvPoint2D32f(predict_value_x, predict_value_y));
}

void kalman::refine(float x, float y)
{
	/* update state with current position */
	cvkalman->state_post->data.fl[0] = x;
	cvkalman->state_post->data.fl[2] = y;

	cvRandSetRange(&rng, 0, sqrt(cvkalman->measurement_noise_cov->data.fl[0]), 0 );
	cvRand(&rng, measurement );

	//cvMatMulAdd( cvkalman->transition_matrix, state, process_noise, cvkalman->state_post );

	cvMatMulAdd(cvkalman->measurement_matrix, cvkalman->state_post, measurement, measurement );

	/* adjust Kalman filter state */
	cvKalmanCorrect(cvkalman, measurement );
	//float measured_value_x = measurement->data.fl[0];
	//float measured_value_y = measurement->data.fl[2];

}

void kalman::init_kalman(int x, int xv, int y,int yv)
{
	cvkalman->state_post->data.fl[0]=x;
	cvkalman->state_post->data.fl[1]=xv;
	cvkalman->state_post->data.fl[2]=y;
	cvkalman->state_post->data.fl[3]=yv;
}

void kalman::release()
{
	cvReleaseKalman(&cvkalman);
	cvReleaseMat(&measurement);
	return;
}


#endif
