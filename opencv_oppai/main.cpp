#include <opencv\cv.h>
#include <opencv\highgui.h>

int main(void){

	// ‘ÎÛ‰æ‘œ
	char fileName[] = "oppai.jpg";
	IplImage* input = cvLoadImage(fileName, 1);

	// ‚¨‚Á‚Ï‚¢ŠçŒŸoŠí‚Ì“Ç‚İ‚İ
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)cvLoad("cascade_oppai.xml");

	CvMemStorage* strage = cvCreateMemStorage(0);
	CvSeq* oppais;
	int i;

	// ‚¨‚Á‚Ï‚¢ŒŸo
	oppais = cvHaarDetectObjects(input, cascade, strage);

	// ‚¨‚Á‚Ï‚¢—Ìˆæ‚Ì•`‰æ
	for(int i=0; i<oppais->total; i++){
		CvRect oppai_rect = *(CvRect*)cvGetSeqElem(oppais, i);
		cvRectangle(
			input,
			cvPoint(oppai_rect.x, oppai_rect.y),
			cvPoint((oppai_rect.x+oppai_rect.width),(oppai_rect.y+oppai_rect.height)),
			CV_RGB(255, 0, 0),
			3
		);
	}

	// •\¦
	cvReleaseMemStorage(&strage);
	cvNamedWindow("oppai_detect", 1);
	cvShowImage("oppai_detect", input);

	cvWaitKey(0);

	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseImage(&input);

	return 0;
}