#include <opencv\cv.h>
#include <opencv\highgui.h>

int main(void){

	// �Ώۉ摜
	char fileName[] = "oppai.jpg";
	IplImage* input = cvLoadImage(fileName, 1);

	// �����ς��猟�o��̓ǂݍ���
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)cvLoad("cascade_oppai.xml");

	CvMemStorage* strage = cvCreateMemStorage(0);
	CvSeq* oppais;
	int i;

	// �����ς����o
	oppais = cvHaarDetectObjects(input, cascade, strage);

	// �����ς��̈�̕`��
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

	// �\��
	cvReleaseMemStorage(&strage);
	cvNamedWindow("oppai_detect", 1);
	cvShowImage("oppai_detect", input);

	cvWaitKey(0);

	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseImage(&input);

	return 0;
}