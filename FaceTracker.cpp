///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions (licence) and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * The name of the author may not be used to endorse or promote products
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through
//       Subspace Constrained Mean-Shifts. International Conference of Computer
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

// Hugely Modified by Md. Iftekhar Tanveer for Blind Emotion Project

///////////////////////////////////////////////////////////////////////////////
// This is our CSC 212/412 Human Computer Interaction Project
// It works as a small face recognition detector that can detect your nod,
// shake and India nod gesture, and your smile and surprise facial expression
//
// Everyone with a Windows computer and a camera can play with it.
//
// Team member:
// 1. Florencia
// 2. Laura
// 3. Mounic
// 4. Jing Sun
// 5. Fangzhou Liu
///////////////////////////////////////////////////////////////////////////////
#include <Tracker.h>
//#include <highgui.h>
#include <iostream>
//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
    int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;
    
    //draw triangulation
    c = CV_RGB(0,0,0);
    for(i = 0; i < tri.rows; i++){
        if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
           visi.at<int>(tri.at<int>(i,1),0) == 0 ||
           visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                       shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                       shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,c);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                       shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                       shape.at<double>(tri.at<int>(i,2)+n,0));
        cv::line(image,p1,p2,c);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                       shape.at<double>(tri.at<int>(i,2)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                       shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,c);
    }
    //draw connections
    c = CV_RGB(0,0,255);
    for(i = 0; i < con.cols; i++){
        if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
           visi.at<int>(con.at<int>(1,i),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
                       shape.at<double>(con.at<int>(0,i)+n,0));
        p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
                       shape.at<double>(con.at<int>(1,i)+n,0));
        cv::line(image,p1,p2,c,1);
    }
    //draw points
    for(i = 0; i < n; i++){
        if(visi.at<int>(i,0) == 0)continue;
        p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
        c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
    }return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv,
              char* ftFile,char* conFile,char* triFile,
              bool &fcheck,double &scale,int &fpd)
{
    int i; fcheck = false; scale = 1; fpd = -1;
    for(i = 1; i < argc; i++){
        if((std::strcmp(argv[i],"-?") == 0) ||
           (std::strcmp(argv[i],"--help") == 0)){
            std::cout << "track_face:- Written by Jason Saragih 2010" << std::endl
            << "Performs automatic face tracking" << std::endl << std::endl
            << "#" << std::endl
            << "# usage: ./face_tracker [options]" << std::endl
            << "#" << std::endl << std::endl
            << "Arguments:" << std::endl
            << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
            << std::endl
            << "-c <string> -> Connectivity (default: ../model/face.con)"
            << std::endl
            << "-t <string> -> Triangulation (default: ../model/face.tri)"
            << std::endl
            << "-s <double> -> Image scaling (default: 1)" << std::endl
            << "-d <int>    -> Frames/detections (default: -1)" << std::endl
            << "--check     -> Check for failure" << std::endl;
            return -1;
        }
    }
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
    }
    if(i >= argc)fcheck = false;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-s") == 0){
            if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
            break;
        }
    }
    if(i >= argc)scale = 1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-d") == 0){
            if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
            break;
        }
    }
    if(i >= argc)fpd = -1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-m") == 0){
            if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
            else strcpy(ftFile,"../model/face2.tracker");
            break;
        }
    }
    if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-c") == 0){
            if(argc > i+1)std::strcpy(conFile,argv[i+1]);
            else strcpy(conFile,"../model/face.con");
            break;
        }
    }
    if(i >= argc)std::strcpy(conFile,"../model/face.con");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-t") == 0){
            if(argc > i+1)std::strcpy(triFile,argv[i+1]);
            else strcpy(triFile,"../model/face.tri");
            break;
        }
    }
    if(i >= argc)std::strcpy(triFile,"../model/face.tri");
    return 0;
}
float getSystemTime(){
    return cv::getTickCount()/cv::getTickFrequency()*1000;
}

// =============== helper func we created for this proj ===============

/*=============================================
 Distance Calculator
 =============================================*/
float Distance(float x1, float x2, float y1, float y2)
{
    float d;
    d = pow((x1 - x2), 2) + pow((y1 - y2), 2);
    return sqrt(d);
}

/*=============================================
 Timer Counter
 =============================================*/
bool isTimerExpired(float startTime)
{
    if (getSystemTime() - startTime < 4000)
    {
        return true;
    }
    return false;
}

/*=============================================
 Mouse Open Detector
 =============================================*/
bool mouseopen(float xb, float yb, float xh, float yh, float ori_distance, int i, int level)
{
    if (Distance(xb, yb, xh, yh) > 1.1*ori_distance)
    {
        if (level == 1)
        {
            printf("\n\nMOUSE OPEN!!!! in at %d times\n\n", i);
        }
        return true;
    }
    else
    {
        return false;
    }
}

/*=============================================
 Eyebrow Raise Detector
 =============================================*/
bool eyebrowraise(float xh, float yh, float xref, float yref, float ori_distance, char dir, int times, int level)
{
    if (Distance(xh, yh, xref, yref) > 1.028*ori_distance)
    {
        if (level == 1)
        {
            printf("\n\nEYE BROW RAISE!!!! in %c at %d times\n\n", dir, times);
        }
        return true;
    }
    else
    {
        return false;
    }
}

//=============================================================================
/*=============================================
 Main Function
 =============================================*/
int main(int argc, const char** argv)
{
    //parse command line arguments
    char ftFile[256],conFile[256],triFile[256];
    bool fcheck = false; double scale = 1; int fpd = -1; bool show = true;
    if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,scale,fpd)<0)return 0;
    
    //set other tracking parameters
    std::vector<int> wSize1(1); wSize1[0] = 7;
    std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 5; double clamp=3,fTol=0.01;
    FACETRACKER::Tracker model(ftFile);
    cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
    cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
    
    //initialize camera and display window
    cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text;
    CvCapture* camera = cvCreateCameraCapture(CV_CAP_ANY); if(!camera)return -1;
    int64 t1,t0 = cvGetTickCount(); int fnum=0;
    cvNamedWindow("Face Tracker",1);
    std::cout << "Hot keys: "        << std::endl
    << "\t ESC - quit"     << std::endl
    << "\t d   - Redetect" << std::endl;
    
    //loop until quit (i.e user presses ESC)
    bool failed = true;
    double pitch = 0, yaw = 0, roll = 0;
    
    int n = model._shape.rows / 2;
    
    // =============== Values we created for NOD/ SHAKE/ INDIA NOD func ===============
    bool isFirst = true;
    bool isUp = false;
    bool isLeft = false, isRight = false;
    bool isRollChange = false;
    float startP = 0.0, startY = 0.0, startR = 0.0;
    float startTime = 0.0;
    
    // =============== Values we created for SMILE & SURPRISE func ===============
    int M_Right_Center = 54;
    int M_Left_Center = 48;
    int M_Top_Center = 51;
    int M_Bottom_Center = 57;
    
    int L_EB_Rightmost = 21;
    int L_EB_Highest = 19;
    int L_EB_Leftmost = 17;
    
    int R_EB_Leftmost = 22;
    int R_EB_Highest = 24;
    int R_EB_Rightmost = 26;
    
    int L_E_Bottom = 41;
    int R_E_Bottom = 46;
    
    int EB_FIXED = 27;
    
    float m_orig_distant_height = 0.0;
    float eb_orig_distant_left = 0.0;
    float eb_orig_distant_right = 0.0;
    
    int loop = 0;
    int level = 0;
    int i = 0;
    double prevdist = 0;
    int loops = 0;
    
    // Main loop
    //bool down = false;
    while(1){
        //grab image, resize and flip
        IplImage* I = cvQueryFrame(camera); if(!I)continue; frame = I;
        if(scale == 1)im = frame;
        else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
        cv::flip(im,im,1); cv::cvtColor(im,gray,CV_BGR2GRAY);
        
        //track this image
        std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1;
        if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
            int idx = model._clm.GetViewIdx(); failed = false;
            
            // Extract pitch, yaw and roll movements
            pitch = model._clm._pglobl.at<double>(1);
            yaw = model._clm._pglobl.at<double>(2);
            roll = model._clm._pglobl.at<double>(3);
            
            // =================================================================
            // =============== Homework: Your code will go here >>
            // printf("time: %f, Pitch = %0.2f  Yaw = %0.2f  Roll = %0.2f\n",getSystemTime(),pitch,yaw,roll);
            
            if (isFirst) {
                startP = pitch;
                startR = roll;
                startY = yaw;
                isFirst = false;
            }
            else {
                // nod
                //printf("\npitch - startP = %.2f\n", pitch - startP);
                /*if (!isUp && (pitch - startP) < -0.03) {
                 //printf("\n UP \n");
                 if (abs(abs(startY) - abs(yaw)) < 0.03 &&  pitch > 0) {
                 isUp = true;
                 startTime = getSystemTime(); // start the time counter
                 }
                 }
                 if (isUp && (pitch - startP) > 0.03) {
                 if (isTimerExpired(startTime)) {
                 if (abs(abs(startY) - abs(yaw)) < 0.03 && roll > 0) {
                 isUp = false;
                 printf("\n\n one YES detected! \n\n");
                 }
                 }
                 else {
                 // time invalid, the head doesn't UP
                 isUp = false;
                 }
                 }*/
                
                // ========================= YES/ NO/ INDIA NO DETECTION ==========================
                
                // no
                if (!isLeft) {
                    printf("\n LEFT \n");
                    if (yaw >= 0.07) {
                        isLeft = true;
                        startTime = getSystemTime();
                    }
                }
                
                if (isLeft && yaw <= -0.07) {
                    if (isTimerExpired(startTime)) {
                        printf("\n\n one NO detected! \n\n");
                    }
                    isLeft = false;
                } else{
                    isLeft = false;
                }
                
                //yes
                //  if (!isUp && !isLeft) {
                if(!(yaw >= 0.10))  {
                    if (pitch <= -0.15) {
                        isUp = true;
                        startTime = getSystemTime();
                    }
                }
                
                if (isUp && pitch >= 0.02) {
                    if (isTimerExpired(startTime)) {
                        printf("\n\none YES detected!\n\n");
                    }
                    isUp = false;
                }
                
                cv::Point nosecenterpt = cv::Point(model._shape.at<double>(EBFixed, 0), model._shape.at<double>(EBFixed + n, 0));
                double x = nosecenterpt.x;
                double y = nosecenterpt.y;
                // printf("nose %f %f\n", x, y);
                
                // Indian Nod
                //  if (!isRollChange) {
                if (roll <= -0.1) {
                    isRollChange = true;
                    startTime = getSystemTime();
                }
                //}
                
                if (isRollChange) {
                    if (roll > 0.2) {
                        if (isTimerExpired(startTime)) {
                            printf("\n\n one INDIA NOD detected! \n\n");
                        }
                        isRollChange = false;
                    }
                }
            }
            
            // ========================= SMILE DETECTION ==========================
            double rightmouthcornerx, rightmouthcornery,leftmouthcornerx, leftmouthcornery;
            
            cv::Point prightcornermouth= cv::Point(model._shape.at<double>(M_Right_Center, 0), model._shape.at<double>(M_Right_Center + n, 0));
            cv::Point pleftcornermouth = cv::Point(model._shape.at<double>(M_Left_Center, 0), model._shape.at<double>(M_Left_Center + n, 0));
            
            if (loops >= 10) {
                double dist = abs(prightcornermouth.x - pleftcornermouth.x);
                if (prevdist != 0) {
                    if (dist > (1.2*prevdist)) {
                        //printf("dist: %f, prevdist: %f\n", dist, prevdist);
                        printf("Smile \n\n");
                    }
                }
                
                prevdist = dist;
                loops = 0;
            } else {
                loops++;
            }
            
            // ======================================= SURPRISE DETECTION ============================================
            
            cv::Point LEBHighest = cv::Point(model._shape.at<double>(L_EB_Highest, 0), model._shape.at<double>(L_EB_Highest + n, 0));
            cv::Point REBHighest = cv::Point(model._shape.at<double>(R_EB_Highest, 0), model._shape.at<double>(R_EB_Highest + n, 0));;
            cv::Point EBFixed = cv::Point(model._shape.at<double>(EB_FIXED, 0), model._shape.at<double>(EB_FIXED + n, 0));
            
            cv::Point MTop = cv::Point(model._shape.at<double>(M_Top_Center, 0), model._shape.at<double>(M_Top_Center + n, 0));
            cv::Point MBottom = cv::Point(model._shape.at<double>(M_Bottom_Center, 0), model._shape.at<double>(M_Bottom_Center + n, 0));
            
            bool isMouseOpen = false;
            bool isEyeBrowRaise = false;
            if (loop == 0)
            {
                m_orig_distant_height = Distance(MTop.x, MBottom.x, MTop.y, MBottom.y);
                eb_orig_distant_left = Distance(LEBHighest.x, EBFixed.x, LEBHighest.y, EBFixed.y);
                eb_orig_distant_right = Distance(REBHighest.x, EBFixed.x, REBHighest.y, EBFixed.y);
                // printf("\nm_orig_distant_height = %f\neb_orig_distant_left = %f\neb_orig_distant_right = %f\n", m_orig_distant_height, eb_orig_distant_left, eb_orig_distant_right);
            }
            if (loop >= 2)
            {
                if (mouseopen(MTop.x, MBottom.x, MTop.y, MBottom.y, m_orig_distant_height, i, level))
                {
                    isMouseOpen = true;
                }
                
                if (eyebrowraise(LEBHighest.x, EBFixed.x, LEBHighest.y, EBFixed.y, eb_orig_distant_left, 'L', i, level) && eyebrowraise(REBHighest.x, EBFixed.x, REBHighest.y, EBFixed.y, eb_orig_distant_right, 'R', i, level)) //&& mouseopen(MTop.x, MBottom.x, MTop.y, MBottom.y, m_orig_distant_height,i)) //&& ))
                {
                    isEyeBrowRaise = true;
                    
                }
                if (isEyeBrowRaise && isMouseOpen)
                {
                    printf("\none Surprise detected!\n");
                }
                loop = 0;
                isEyeBrowRaise = false;
                isMouseOpen = false;
            }
            else {
                loop++;
            }
            
            // =================================================================
            
            Draw(im,model._shape,con,tri,model._clm._visi[idx]); 
        }else{
            if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
            model.FrameReset(); failed = true;
        }     
        //draw framerate on display image 
        if(fnum >= 9){      
            t1 = cvGetTickCount();
            fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
            t0 = t1; fnum = 0;
        }else fnum += 1;
        if(show){
            sprintf(sss,"%d frames/sec",(int)ceil(fps)); text = sss;
            cv::putText(im,text,cv::Point(10,20),
                        CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
        }
        //show image and check for user input
        imshow("Face Tracker",im); 
        int c = cvWaitKey(5);
        if(c == 27)
            break; 
        else 
            if(char(c) == 'd')model.FrameReset();
    } // End of main loop
    
    return 0;
}
//=============================================================================
