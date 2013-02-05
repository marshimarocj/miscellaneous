#ifndef IMAGE_INDEX_H
#define IMAGE_INDEX_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include "Image_Feature.h"
#include "Util.h"
#include "Math_Operation.h"
#include "Apriori.h"

using namespace std;

#ifdef WINDOWS
const char directoryChar = '\\';
#else
const char directoryChar = '/';
#endif

const int minblocksize = 16;
const int maxblocks = 8;
const CvSize imageresize = cvSize(minblocksize * maxblocks, minblocksize * maxblocks);

class SuperPixel
{
    public:
        /** The srcImage path */
        string srcImagePath;

        /** This super pixel's region in the srcImage */
        CvRect region;

        /** SuperPixel size */
        int blocksize;

        /** SuperPixel HashCode */
        long long hashcode;

        bool operator< (const SuperPixel & other) const
        {
            return hashcode < other.hashcode;
        }


    public:
        ///////////////////////////////////////////////////////////
        // Construct
        ///////////////////////////////////////////////////////////
        SuperPixel(string & srcImage, CvRect & rect)
        {
            srcImagePath = srcImage;
            region = rect;
            blocksize = rect.width;

            IplImage * img = getImage();
            hashcode = getImageHashCode(img);
            //feature = Feature::factory("mix");
            //feature->init(img);
            cvReleaseImage(&img);
        }

        SuperPixel(string & srcImage, CvRect & rect, long long code)
        {
            srcImagePath = srcImage;
            region = rect;
            blocksize = rect.width;
            hashcode = code;
        }

        /**
        ~SuperPixel()
        {
            if (feature != NULL) 
                delete feature;
        }
        */

        //////////////////////////////////////////////////////////////////
        // Return the visual image of this super pixel
        //////////////////////////////////////////////////////////////////
        IplImage * getImage()
        {
            IplImage * srcImage = cvLoadImage(srcImagePath.c_str());
            IplImage * resizeImage = cvCreateImage(imageresize,8,3);
            cvResize(srcImage,resizeImage);
            IplImage * ret = getPartialImage(resizeImage,region);
            cvReleaseImage(&srcImage);
            cvReleaseImage(&resizeImage);
            return ret;
        }

        //////////////////////////////////////////////////////////////////
        // Given a list of superpixel , return its visual represent
        // Only display first 20 superpixel
        //////////////////////////////////////////////////////////////////
        static IplImage * getImage(vector<SuperPixel> pixels)
        {
            int size = pixels.size();
            if (size >= 20) size = 20;
            vector<IplImage *> images;
            for (int i = 0; i < size; ++i)
                images.push_back(pixels[i].getImage());
    
            IplImage * ret = combineImageRow(images);

            for (int i = 0; i < size; ++i) cvReleaseImage(&images[i]);
            return ret;
        }

        ///////////////////////////////////////////////////////////////////
        // Given a list of superpixel , return its visual represent
        // 20 per line
        ///////////////////////////////////////////////////////////////////
        static IplImage * getAllImage(vector<SuperPixel> pixels)
        {
            int rows = (pixels.size() - 1) / 20 + 1;

            vector<IplImage *> images;
            for (int i = 0; i < rows; ++i) {
                int upperBound = min((int)pixels.size() , 20 * i + 20);
                vector<SuperPixel> now;
                for (int j = i * 20; j < upperBound; ++j)
                    now.push_back(pixels[j]);
                images.push_back(getImage(now));
            }

            IplImage * ret = combineImage(images);
            for (int i = 0; i < images.size(); ++i)
                cvReleaseImage(&images[i]);
            return ret;
        }

        static string toString(const set<SuperPixel>  & pixels)
        {
            string ret = "";
            char temp[500];
            for (set<SuperPixel>::iterator itr = pixels.begin(); itr !=
                    pixels.end(); ++itr) {
                sprintf(temp, "%lld", itr->hashcode);
                ret += string("_") + temp;
            }
            return ret;
        }

        string toString()
        {
            string ret = "";
            ret += srcImagePath + "\n";
            char temp[100];
            sprintf(temp, "%d\t%d\t%d\t%d", region.x, region.y, region.width,
                    region.height);
            ret += string(temp);
            return ret;
        }
};

///////////////////////////////////////////////////
// Code Book Generator
///////////////////////////////////////////////////
class CodeBookGenerator
{
    public:
        /** images with the given text discription */
        map<string, set<string> > textToRelatedImage;

        /** superpixels in the given image */
        map<string, set<SuperPixel> > imageToContainedSuperPixel;

        /** hashcode to superpixels */
        map<long long, vector<SuperPixel> > hashcodeToSuperPixel;

        /** image with color-tone */
        map<string, string> imageToColorTone;

        /** Appeared Color Tone */
        set<string> colorTones;

        /** Split to srcImage to rowblocks * rowblocks area */
        int rowblocks;

        /** Each SuperPixel's size is blocksize * blocksize */
        int blocksize;

        /** left corner coodinate distance between neighbour superpixels */
        int delta;

    public:
        CodeBookGenerator(int _rowblocks, int _blocksize, int _delta)
        {
            rowblocks = _rowblocks;
            blocksize = _blocksize;
            delta = _delta;
        }

        void clear()
        {
            this->textToRelatedImage.clear();
            this->imageToContainedSuperPixel.clear();
            this->hashcodeToSuperPixel.clear();
            this->imageToColorTone.clear();
            this->colorTones.clear();
        }

        //////////////////////////////////////////////////////////////////////////////
        // Given a set of hashcodes, return its visual represent 
        //////////////////////////////////////////////////////////////////////////////
        IplImage * getHashCodesDisplayImage(const set<SuperPixel> & hashcodes)
        {
            vector<IplImage *> images;
            set<SuperPixel>::iterator itr;
            for (itr = hashcodes.begin(); itr != hashcodes.end(); ++itr) {
                long long hashcode = (*itr).hashcode;
                if (hashcodeToSuperPixel.find(hashcode) == hashcodeToSuperPixel.end())
                    continue;
                vector<SuperPixel> & pixels = hashcodeToSuperPixel[hashcode];
                IplImage * img = SuperPixel::getImage(pixels);
                images.push_back(img);
            }

            IplImage * ret = combineImage(images);
            for (int i = 0; i < images.size(); ++i) 
                if (images[i] != NULL) cvReleaseImage(&images[i]);
            
            return ret;
        }

        //////////////////////////////////////////////////////
        // Given an image with some annotated text 
        // Insert to the index
        /////////////////////////////////////////////////////
        int insert(string img, vector<string> words, bool calSalientRegion = false)
        {
            IplImage * srcImage = cvLoadImage(img.c_str());
            if (srcImage == NULL) return 0;

            IplImage * resizeImage = cvCreateImage(imageresize,8,3);
            cvResize(srcImage,resizeImage);

            /** text to related images updating */
            for (int i = 0; i < words.size(); ++i) {
                if (textToRelatedImage.find(words[i]) == textToRelatedImage.end()) {
                    set<string> imageNameSet;
                    textToRelatedImage[words[i]] = imageNameSet;
                }
                textToRelatedImage[words[i]].insert(img);
            }

            CvMat * salient;
            /** If need salient region , cal it */
            if (calSalientRegion) {
                IplImage * seg;
                IplImage * salientMap;
                IplImage * salientImage;
                vector<pair<int, int> > salientBlocks;
                salient = getImageSalientRegion(resizeImage, seg, salientMap,
                        salientImage, salientBlocks);
                cvReleaseImage(&seg);
                cvReleaseImage(&salientMap);
                cvReleaseImage(&salientImage);
            }

            /** image to contained superpixel updating */
            set<SuperPixel> superPixels;
            vector<SuperPixel> vSuperPixels;
            vector<long long> codes;
            vector<CvRect> rects;
            for (int i = 0; i < rowblocks; ++i)
                for (int j = 0; j < rowblocks; ++j) {
                    CvRect rect;
                    rect.x = i * delta;
                    rect.y = j * delta;
                    rect.height = blocksize;
                    rect.width = blocksize;

                    if (calSalientRegion) {
                        int count = 0;
                        for (int x = rect.x; x < rect.x + rect.height; ++x)
                            for (int y = rect.y; y < rect.y + rect.width; ++y)
                                if (cvmGet(salient, x, y) > 0) 
                                    count++;
                        if (count <= rect.height * rect.width / 4) 
                            continue;
                    }
                    rects.push_back(rect);
                }

            if (calSalientRegion) 
                cvReleaseMat(&salient);

            for (int i = 0; i < rects.size(); ++i) {
                IplImage * superpixelImage = getPartialImage(resizeImage,
                        rects[i]);
                long long code = getImageHashCode(superpixelImage);
                codes.push_back(code);
                SuperPixel superpixel(img, rects[i], codes[i]);
                superPixels.insert(superpixel);
                vSuperPixels.push_back(superpixel);
                cvReleaseImage(&superpixelImage);
            }

            imageToContainedSuperPixel[img] = superPixels;

            /** hashcode to superpixel updating */
            for (int i = 0; i < codes.size(); ++i) {
                if (hashcodeToSuperPixel.find(codes[i]) == 
                        hashcodeToSuperPixel.end()) {
                    vector<SuperPixel> superPixels;
                    hashcodeToSuperPixel[codes[i]] = superPixels;
                }
                hashcodeToSuperPixel[codes[i]].push_back(vSuperPixels[i]);
            }

            cvReleaseImage(&srcImage);
            cvReleaseImage(&resizeImage);
            return 1;
        }

        /** Load a directory's images related to a word */
        int loadSingleWordImages(string word, string path)
        {
            int training = 0;
            vector<string> words;
            words.push_back(word);

            FileFinder f(path.data());
            while (f.hasNext())
            {
                training += this->insert(f.next(),words);
            }
            return training;
        }

        /** Load a directory's images related to a word, 
         *  Only load a given color tone 
         */
        int loadSingleWordImages(string word, string path, string colorTone)
        {
            int training = 0;
            vector<string> words;
            words.push_back(word);

            FileFinder f(path.data());
            while (f.hasNext())
            {
                string filename = f.next();
                IplImage * img = cvLoadImage(filename.data());
                if (img == NULL) continue;
                string nowColor = Feature::getImageColorTone(img);
                cvReleaseImage(&img);
                if (nowColor == colorTone)
                    training += this->insert(filename, words);
            }
            cout << colorTone << "\t" << training << endl;
            return training;
        }

        ////////////////////////////////////////////////////////////
        // Generate Image Code Book By different Color-Tone
        ////////////////////////////////////////////////////////////
        int generateCodeBook(string word, string path, 
                double minsupportRadio, string savepath, string codeFeatureFile, 
                bool needSalientRegion = false)
        {
            Util::mkdir(savepath);
            vector<string> colorTones = Feature::getImageColorTones();
            vector<pair<string, string> > images;
            vector<string> words;
            words.push_back(word);

            FileFinder f(path.data());
            while (f.hasNext()) {
                string filename = f.next();
                IplImage * img = cvLoadImage(filename.data());
                if (img == NULL) continue;
                string nowColor = Feature::getImageColorTone(img);
                cvReleaseImage(&img);
                images.push_back(make_pair(filename, nowColor));
                cout << filename << "\t" << nowColor << endl;
            }

            for (int i = 0; i < colorTones.size(); ++i) {
                string nowColor = colorTones[i];
                cout << "Now Color Tone = " << nowColor << endl;
                this->clear();
                int count = 0;
                for (int j = 0; j < images.size(); ++j) 
                    if (images[j].second == nowColor) {
                        this->insert(images[j].first, words, needSalientRegion);
                        count++;
                    }
                if (count >= 4)
                    generateCodeBook(word, minsupportRadio, savepath, codeFeatureFile, nowColor);
            }
        }

        /////////////////////////////////////////////////////////////
        // Generate Image Code Book
        /////////////////////////////////////////////////////////////
        int generateCodeBook(string word, double minsupportRadio, 
               string savepath, string codeFeatureFile, string nowColor = "")
        {
            ofstream fout(codeFeatureFile.data(), ofstream::app);

            if (savepath.data()[savepath.length()-1] != directoryChar) 
                savepath += directoryChar;
            if (textToRelatedImage.find(word) == textToRelatedImage.end()) 
                return 0;

            Util::mkdir(savepath + "1");
            Util::mkdir(savepath + "2");
            Util::mkdir(savepath + "3");
            Util::mkdir(savepath + "4");
            Util::mkdir(savepath + "5");

            Apriori<SuperPixel> apriori;
            vector<set<SuperPixel> > itemList;

            /** Init to call apriori */
            set<string> & files = textToRelatedImage[word];

            itemList.clear();
            for (set<string>::iterator itr = files.begin(); itr != files.end(); ++itr) {
                const set<SuperPixel> & pixels = imageToContainedSuperPixel[*itr];
                itemList.push_back(pixels);
            }

            /** Call Apriori Algorithm */
            vector<set<set<SuperPixel> > > result = 
                apriori.miningFrequentPattern(itemList, minsupportRadio, 5, 30);

            map<long long, int> codeBookCode;
            char filename[10000];
            for (int i = 0; i < result.size(); ++i) {
                cout << i + 1 << " Frequent Item = " << result[i].size() << endl;

                set<set<SuperPixel> > & frequent = result[i];
                set<set<SuperPixel> >::iterator itr;
                for (itr = frequent.begin(); itr != frequent.end(); ++itr) {
                    const set<SuperPixel> & frequentItem = (*itr);

                    /** Add these code to codebook */
                    for (set<SuperPixel>::iterator itr = frequentItem.begin();
                            itr != frequentItem.end(); ++itr)
                        codeBookCode[itr->hashcode] = i + 1;

                    /** Output Images */
                    IplImage * nowImage =
                        this->getHashCodesDisplayImage(frequentItem);
                    sprintf(filename, "%s%d%c%s_%s%s.jpg", savepath.data(), i + 1,
                            directoryChar, word.data(), nowColor.data(),
                            SuperPixel::toString(frequentItem).data());
                    cout << filename << endl;
                    cvSaveImage(filename, nowImage);
                    cvReleaseImage(&nowImage);
                }
            }

            cout << "Save Code Book" << endl;
            for (map<long long, int>::iterator itr = codeBookCode.begin();
                    itr != codeBookCode.end(); ++itr) {
                vector<SuperPixel> & pixels = hashcodeToSuperPixel[itr->first];

                for (int i = 0; i < pixels.size(); ++i) {
                    fout << word << "\t" << nowColor << "\t" << pixels[i].toString() << endl;
                    IplImage * smallImage = pixels[i].getImage();
                    Feature * feature = Feature::factory("mix");
                    feature->init(smallImage);
                    cvReleaseImage(&smallImage);
                    fout << (*feature) << endl;
                }

                IplImage * nowImage = SuperPixel::getAllImage(pixels);
                sprintf(filename, "%s%s_%s_%d_%lld.jpg",
                        savepath.data(), nowColor.data(), word.data(), itr->second, itr->first);
                cout << filename << endl;
                cvSaveImage(filename, nowImage);
                cvReleaseImage(&nowImage);
            }


            fout.close();
        }
        };

#endif

