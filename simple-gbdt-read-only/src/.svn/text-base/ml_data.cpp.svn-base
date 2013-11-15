#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <errno.h>

#include "ml_data.h"


static void InplaceTrimLeft(std::string& strValue)
{
	size_t pos = 0;
	for (size_t i = 0; i < strValue.size(); ++i)
	{
		if (isspace((unsigned char)strValue[i]))
			++pos;
		else
			break;
	}
	if (pos > 0)
		strValue.erase(0, pos);
}

static void InplaceTrimRight(std::string& strValue)
{
	size_t n = 0;
	for (size_t i = 0; i < strValue.size(); ++i)
	{
		if (isspace((unsigned char)strValue[strValue.length() - i - 1]))
			++n;
		else
			break;
	}
	if (n != 0)
		strValue.erase(strValue.length() - n);
}

static void InplaceTrim(std::string& strValue)
{
	InplaceTrimRight(strValue);
	InplaceTrimLeft(strValue);
}

static void Split(
	const std::string& strMain,
	char chSpliter,
	std::vector<std::string>& strList,
	bool bReserveNullString)
{
	strList.clear();

	if (strMain.empty())
		return;

	size_t nPrevPos = 0;
	size_t nPos;
	std::string strTemp;
	while ((nPos = strMain.find(chSpliter, nPrevPos)) != std::string::npos)
	{
		strTemp.assign(strMain, nPrevPos, nPos - nPrevPos);
		InplaceTrim(strTemp);
		if (bReserveNullString || !strTemp.empty())
			strList.push_back(strTemp);
		nPrevPos = nPos + 1;
	}

	strTemp.assign(strMain, nPrevPos, strMain.length() - nPrevPos);
	InplaceTrim(strTemp);
	if (bReserveNullString || !strTemp.empty())
		strList.push_back(strTemp);
}

bool DataReader::ReadDataFromCVS(const std::string& input_file, Data& data)
{
    std::ifstream fs;
    fs.open(input_file.c_str(), std::ios_base::in);

    if (fs.fail())
    {
        std::cerr << " Sorry ! The file isn't exist. " << input_file << std::endl;
        return false;
    }

    std::string strLine;
    unsigned int line_num = 0;
    while (getline(fs, strLine))
    {
        if (!strLine.empty())
        {
            std::vector<std::string> vecResult;
            Split(strLine, ',', vecResult,true);
            if (vecResult.size() >= 2)
            {
                data.m_data.resize( line_num + 1 );
                T_VECTOR & fv = data.m_data[line_num];
                
                T_DTYPE f_value = 0;
                for ( size_t i = 0; i < vecResult.size() - 1; ++i )
                {
                    char ** endptr = NULL;
                    f_value = strtof( vecResult[i].c_str() , endptr);
                    if(errno == ERANGE)
                    {
                          std::cerr << " feature out of the range: " << vecResult[i] << std::endl;
                    }
                    if(endptr != NULL && endptr[0] == vecResult[i].c_str())
                    {
                          std::cerr << " feature format wrong: " << vecResult[i] << std::endl;
                    }

                    fv.push_back( f_value );
                }
                
                unsigned int target_index = vecResult.size() - 1;
                char ** endptr = NULL;
                f_value = strtof( vecResult[target_index].c_str() , endptr);
                if(errno == ERANGE)
                {
                      std::cerr << " target out of the range: " << vecResult[target_index] << std::endl;
                }
                if(endptr != NULL && endptr[0] == vecResult[target_index].c_str())
                {
                      std::cerr << " target format wrong: " << vecResult[target_index] << std::endl;
                }
                data.m_target.push_back(f_value);
                
                if (line_num == 0)
                {
                    data.m_dimension = data.m_data[0].size();
                }
                line_num++;
            }
            else
            {
                //TODO statistic wrong line
            }
        }
    }
    if ( data.m_data.size() > 0 )
    {
        data.m_num = data.m_data.size();
    }
    else
    {
        return false;
    }

    std::cout << "dimension: " << data.m_dimension << std::endl;
    std::cout << "data num: " << data.m_num << std::endl;
    
    return true;
}


bool DataReader::ReadDataFromL2R(const std::string& input_file, Data& data, unsigned int dimentions)
{
    std::ifstream fs;
    fs.open(input_file.c_str(), std::ios_base::in);

    if (fs.fail())
    {
        std::cerr << " Sorry ! The file isn't exist. " << input_file << std::endl;
        return false;
    }

    data.m_dimension = dimentions;
    std::string strLine;
    unsigned int line_num = 0;
    while (getline(fs, strLine))
    {
        if (!strLine.empty())
        {
            std::vector<std::string> vecResult;
            Split(strLine, '\t', vecResult,true);
            if (vecResult.size() >= 3)
            {
                data.m_data.resize( line_num + 1 );
                T_VECTOR & fv = data.m_data[line_num];
                fv.resize(dimentions);
                
                //read target
                unsigned int target_index = 0;
                char ** endptr = NULL;
                T_DTYPE target_value = strtof( vecResult[target_index].c_str() , endptr);
                if(errno == ERANGE)
                {
                      std::cerr << " target out of the range: " << vecResult[target_index] << std::endl;
                      continue;
                }
                if(endptr != NULL && endptr[0] == vecResult[target_index].c_str())
                {
                      std::cerr << " target format wrong: " << vecResult[target_index] << std::endl;
                      continue;
                }
                data.m_target.push_back(target_value);

                
                for ( size_t i = 2; i < vecResult.size(); ++i )
                {
                    int f_index = -1;
                    T_DTYPE f_value = 0.0;
                    int ret = sscanf( vecResult[i].c_str(), "%d:%f", &f_index, &f_value);
                    if (ret != 2 || f_index >= (int)dimentions)
                    {
                        std::cerr << " feature format wrong: " << line_num << "\t"
                            << vecResult[i] << std::endl;
                        continue;
                    }
                    fv[f_index] = f_value;
                    data.m_valid_id.insert(f_index);
                }
                
                line_num++;
            }
            else
            {
                //TODO statistic wrong line
            }
        }
    }
    if ( data.m_data.size() > 0 )
    {
        data.m_num = data.m_data.size();
    }
    else
    {
        return false;
    }

    std::cout << "dimension: " << data.m_dimension << std::endl;
    std::cout << "data num: " << data.m_num << std::endl;
    std::cout << "valid feature size: " << data.m_valid_id.size() << std::endl;
    
    return true;
}

