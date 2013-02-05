#ifndef UTIL_H
#define UTIL_H

#include "stdafx.h"
#ifdef WINDOWS
#include "windows.h"
const char DirectoryChar = '\\';
#else
const char DirectoryChar = '/';
#endif

using namespace std;

class TimeUtil
{
public:

  map<string, time_t> start;
  map<string, time_t> stop;

  void startCount(const string & type)
  {
    start[type] = clock();
  }

  void stopCount(const string & type)
  {
    stop[type] = clock();
    cout << type << "\t" << (stop[type] - start[type]) / 1000.0 << endl;
  }
};

/** A class to find given type files in the given directory */
class FileFinder
{
public:
    string dir;
    vector<string> files;
    unsigned int nowposi;

  public:
    FileFinder(string _dir)
    {
      static string filelist = "pig.filelist.dat";

      dir = _dir;
      if (dir.data()[dir.length()-1] != DirectoryChar) 
        dir += string(1, DirectoryChar);

      files.clear();
      nowposi = 0;

      char command[1000];
#ifdef WINDOWS
      sprintf(command, "dir %s /B > %s",dir.data() , filelist.data());
#else
      sprintf(command, "ls -l %s | awk '/^-/ { print $9}' > %s" ,
          dir.data() , filelist.data());
#endif
	  cout << command << endl;
      system(command);

      /** read the file list in the file */
      ifstream fin(filelist.data());
      string filename;
      while (getline(fin, filename))
        files.push_back(filename);
	  sort(files.begin(), files.end());
      fin.close();
    }

    bool hasNext()
    {
      return nowposi < files.size();
    }

    string next()
    {
      return dir + files[nowposi++];
    }

    string getLastFileName()
    {
      return files[nowposi-1];
    }

    /** Static Function , get the files in the dir with given type */
    static vector<string> getFiles(string dir, string filetype) {
      char command[1000];
      sprintf(command, "find %s | grep -E '.*\\.(%s)$' > pig.filelist.dat",
          dir.data(), filetype.data());
      system(command);

      vector<string> ret;
      ifstream fin("pig.filelist.dat");
      string line;
      while (getline(fin, line)) {
        ret.push_back(line);
      }
      return ret;
    }
};

class DirFinder
{
  private:
    string dir;
    vector<string> dirs;
    unsigned int nowposi;

  public:
    DirFinder(string _dir)
    {
      static string dirlist = "pig.dirlist.dat";

      dir = _dir;
      if (dir.data()[dir.length()-1] != DirectoryChar) 
        dir += string(1, DirectoryChar);

      dirs.clear();
      nowposi = 0;

      char command[1000];
#ifdef WINDOWS
      sprintf(command, "dir %s /B > %s",dir.data() , dirlist.data());
#else
      sprintf(command, "ls -l %s | awk '/^d/ { print $9}' > %s" ,
          dir.data() , dirlist.data());
#endif
      system(command);

      /** read the file list in the file */
      ifstream fin(dirlist.data());
      string filename;
      while (getline(fin, filename))
        dirs.push_back(filename);
      fin.close();
    }

    bool hasNext()
    {
      return nowposi < dirs.size();
    }

    string next()
    {
      return dir + dirs[nowposi++];
    }

    string getLastFileName()
    {
      return dirs[nowposi-1];
    }
};


class Util
{
  public:
    //////////////////////////////////////////////////////////////////////////////////////
    // Make a directory 
    //////////////////////////////////////////////////////////////////////////////////////
    static void mkdir(const char * dir)
    {
      char commandMD[400];
#ifdef WINDOWS
      sprintf(commandMD,"md %s",dir);
#else
      sprintf(commandMD, "mkdir %s" , dir);
#endif
      system(commandMD);
    }

    static void mkdir(const string & dir)
    {
      mkdir(dir.data());
    }

    static void rmdir(const string & dir)
    {
      char command[400];
      sprintf(command, "rmdir %s", dir.data());
      system(command);
    }

	static void copyFile(const string & src, const string & dst)
	{
		char command[400];
		sprintf(command, "copy %s %s /Y", src.data(), dst.data());
		system(command);
	}

  static void moveFile(const string & src, const string & dst)
  {
    char command[400];
    sprintf(command, "move %s %s", src.data(), dst.data());
    //cout << command << endl;
		system(command);
  }

  /**
	static int getFileLength(const string & src)
	{
		ifstream file(src.data());
		if (file.is_open())
		{
			fstream::pos_type cur_pos = file.tellg();
			file.seekg( 0L, ios::end );
			fstream::pos_type end_pos = file.tellg();
			file.seekg( cur_pos, ios::beg );
			return (int) end_pos.seekpos(); 
		}
		else
			return 0;
	}
   */

    /////////////////////////////////////////////////////////
    // D:\Exper\1.jpg
    // 1.jpg
    /////////////////////////////////////////////////////////
    static string getFileName(const string & path)
    {
      int start = path.find_last_of(DirectoryChar);
      return path.substr(start+1,path.length() - start -1);
    }

    //////////////////////////////////////////////////////////
    // D:\Exper\1.jpg
    // D:\Exper\
    ///////////////////////////////////////////////////////////
    static string getFilePath(const string & path)
    {
      int stop = path.find_last_of(DirectoryChar);
      return path.substr(0,stop+1);
    }

    /////////////////////////////////////////////
    // D:\Exper\1.jpg
    // D:\Exper\1
    /////////////////////////////////////////////
    static string getFileWithOutExt(const string & path)
    {
      int posi = path.find_last_of(".");
      return path.substr(0,posi);
    }

    //////////////////////////////////////////////
    // D:\Exper\123.jpg
    // 123
    //////////////////////////////////////////////
    static string getFileTrueName(const string & path)
    {
      return getFileWithOutExt(getFileName(path));
    }

    ///////////////////////////////////////////
    // D:\Exper\123.jpg
    // Exper
    ///////////////////////////////////////////
    static string getFatherPathName(const string & path)
    {
      int stop = path.find_last_of(DirectoryChar);
      string sub = path.substr(0,stop);
      int start = sub.find_last_of(DirectoryChar);
      return sub.substr(start+1,sub.length()-start-1);
    }

    ////////////////////////////////////////////
    // Read the file's whole data as a string 
    ///////////////////////////////////////////
    static string readFile(const string & path)
    {
      ifstream fin(path.data());
      string temp;
      string ret = "";
      if (!fin.fail())
        while (getline(fin , temp)) {
          ret += string(temp) + "\n";
        }
      fin.close();
      return ret;
    }

    static string readURL(const string & url)
    {
      getFile(url, "./pig.html.tmp");
      return readFile("./pig.html.tmp");
    }

    /////////////////////////////////////////
    // Get the file from internet
    // Save to the given path
    /////////////////////////////////////////
    static void getFile(const string & url, const string & savePath)
    {
#ifdef WINDOWS
#else
      string command = "";
      command += "wget ";
      command += " -O ";
      command += savePath;
      command += " -q ";
      command += url;

      system(command.data());
#endif
    }

    static string getTime()
    {
      time_t now;
      time(&now);
      return ctime(&now);
    }

    static bool existFile(const char * filename)
    {
      ifstream f(filename);
      if (f) {   /*如果f不为NULL，说明文件存在，打开成功*/
        f.close();
        return true;
      }
      return false;
    }

    static void run(string cmd, string arg)
    {
      char temp[1000];
      sprintf(temp, "%s %s", cmd.data(), arg.data());
      system(temp);
    }

};

class StringOperation
{
  public:
    /** Split the given string by delimit 
    */
    static vector<string> split(const string & src, const string & delimit, 
        const string &  null_subst = "")
    {    
      if(src.empty() || delimit.empty()) 
        return vector<string>();

      vector<string> v; 
      size_t deli_len = delimit.size();    
      long index = string::npos, last_search_position = 0;    
      while((index = src.find(delimit, last_search_position)) != string::npos)    
      {    
        if(index == last_search_position)    
          v.push_back(null_subst);    
        else   
          v.push_back(src.substr(last_search_position , 
                index - last_search_position));    
        last_search_position = index + deli_len;    
      }    
      string last_one = src.substr(last_search_position);    
      v.push_back(last_one.empty() ? null_subst : last_one);    
      return v;    
    }

	static vector<string> split(const string & src, vector<char> & delimit)
	{
		vector<string> ret;
		int last = 0;
		for (int i = 0; i < src.length(); ++i)
		{
			bool isDelimit = false;
			for (int j = 0; j < delimit.size(); ++j)
				if (src[i] == delimit[j])
					isDelimit = true;
			if (isDelimit)
			{
				ret.push_back(src.substr(last, i - last));
				last = i + 1;
			}
		}
		ret.push_back(src.substr(last, src.length() - last));
		return ret;
	}

    /** Remove the empty character in two ends of string ,return the
     * modified string */
    static string trim(const string& s)
    {
      int i , j;
      for (i = 0; i < s.length(); ++i)
        if (s[i] != ' ' && s[i] != '\t') break;

      for (j = s.length() - 1; j >= 0; --j)
        if (s[j] != ' ' && s[j] != '\t') break;

      return s.substr(i , j - i + 1);
    }

	string Replace( const string& orignStr, const 
	string& oldStr, const string& newStr )   
	{   
		size_t pos = 0;   
		string tempStr = orignStr;   
		string::size_type newStrLen = newStr.length();   
		string::size_type oldStrLen = oldStr.length();   
		while(true)   
		{   
			pos = tempStr.find(oldStr, pos);   
			if (pos == wstring::npos) break;
			tempStr.replace(pos, oldStrLen, newStr);   
			pos += newStrLen;  
		}   
		return tempStr;   
	} 


    static string tolower(const string& s)
    {
      string ret(s);
      for (int i = 0; i < ret.length(); ++i)
        if (ret[i] >= 'A' && ret[i] <= 'Z')
          ret[i] = ret[i] - 'A' + 'a';
      return ret;
    }


    static string toupper(const string & s)
    {
      string ret(s);
      for (int i = 0; i < ret.length(); ++i)
        if (ret[i] >= 'a' && ret[i] <= 'z')
          ret[i] = ret[i] - 'a' + 'A';
      return ret;
    }

    /** Get all the substring between string start and stop */
    static vector<string> getAllMidString(const string & input, const string & start, const string & stop, bool containOther = false)
    {
      vector<string> result;

      size_t startLen = start.length();
      size_t stopLen = stop.length();
      size_t nowstart = 0;
      size_t nextstart = 0;
      while (nextstart < input.length()) {
        nowstart = input.find(start, nextstart);
        if (containOther) {
          if (nowstart == string::npos) 
            nowstart = input.length();
          result.push_back(input.substr(nextstart, nowstart - nextstart));
        }
        if (nowstart == string::npos) break;
        size_t datastart = nowstart + startLen;
        size_t datastop = input.find(stop, datastart);
        if (datastop == string::npos) break;
        string data = input.substr(datastart, datastop - datastart);
        result.push_back(data);
        nextstart = datastop + stopLen;
      }
      return result;
    }

    /** Whether a string is a biaodian or not */
    static bool isDots(const string & str)
    {
      static string chiDots[] = {
        "。", "，", "；", "：", "？", "！", "……", "—",
        "～", "〔", "〕",
        "《", "》", "‘", "’", "“", "”", "`", "．",
        "【", "】"
      };

      static string engDots[] = {
        ".", ",", ";", ":", "?", "!", "…", "-",
        "~", "(", ")",
        "<", ">", "'", "'", "\"", "\"", "'", ".",
        "[", "]"
      };
      static set<string> dots;
      static bool first = true;
      if (first) {
        dots.insert(chiDots, chiDots + 21);
        dots.insert(engDots, engDots + 21);
        first = false;
      }

      if (dots.find(str) != dots.end()) return true;
      return false;
    }


    /** Remove useless char */
    static string getShrinkString(const string & input) 
    {
      int i;
      string ret = input;
      for (i = 0; i < ret.size(); ++i)
        if (ret[i] == '\t') ret[i] = ' ';
      string result = "";
      vector<string> words = split(input, " ");
      for (i = 0; i < words.size(); ++i) 
        if (words[i] != "" && !isDots(words[i]))
          result += words[i];
      return result;
    }

    string getURLHost(const string & url)
    {
      int startPosi = url.find("://");
      int stopPosi = url.find("/" , startPosi + 3);
      if (startPosi == string::npos || stopPosi == string::npos) return url;
      string ret = url.substr(startPosi + 3 , stopPosi - startPosi - 3);
      return ret;
    }

    static inline bool isDigit(const string & str) {
      if (str == ".") return false;
      for (int i = 0; i < str.length(); ++i)
        if (str[i] >= '0' && str[i] <= '9' || str[i] == '.') 
          ;
        else
          return false;
      return true;
    }

    static inline bool isSpecialCase(const string & str) {
      if (str == ".") return false;
      for (int i = 0; i < str.length(); ++i)
        if (str[i] >= 'A' && str[i] <= 'Z' || str[i] == '.')
          ;
        else
          return false;
      return true;
    }

    static inline bool isWord(const string & str) {
      for (int i = 0; i < str.length(); ++i)
        if (str[i] >= 'a' && str[i] <= 'z' || str[i] >= 'A' && str[i]
            <= 'Z')
          ;
        else
          return false;
      return true;
    }

    static int toInt(string s)
    {
      int x;
      sscanf(s.data(), "%d", &x);
      return x;
    }

    static string toString(int x)
    {
      char temp[100];
      sprintf(temp, "%d", x);
      return string(temp);
    }

    static string toString(double x) 
    {
      char temp[100];
      sprintf(temp, "%.2lf", x);
      return string(temp);
    }

    static string to01String(unsigned char c)
    {
      char temp[8];
      for (int i = 7; i >= 0; --i) {
        temp[i] = (c % 2 == 0) ? '0' : '1';
        c /= 2;
      }
      return string(temp);
    }
    
    static string to01String(int x)
    {
      unsigned char * address = (unsigned char *)&x;
      string ret = "";
      for (int i = 3; i >= 0; --i) 
        ret += to01String(*(address + i));
      return ret;
    }

    static string to01String(long x)
    {
      unsigned char * address = (unsigned char *)&x;
      string ret = "";
      for (int i = 7; i >= 0; --i)
        ret += to01String(*(address + i));
      return ret;
    }
};

/**
  int main()
  {
  FileFinder f("/home/pigoneand/Desktop");
  while (f.hasNext()) {
  string file = f.next();
  cout << file << endl;
  }

  return 0;
  }
  */

#endif
