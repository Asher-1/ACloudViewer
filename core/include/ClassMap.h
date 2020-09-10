#pragma once

#include <map>
#include <vector>
#include <string>

namespace ClassMap {
	static std::map<size_t, std::string> SemanticMap = {
			{0 , "Unlabeled" },
			{1 , "Manmade-Terrain" },
			{2 , "Natural-Terrain" },
			{3 , "High-Vegetation" },
			{4 , "Low-Vegetation" },
			{5 , "Buildings" },
			{6 , "Hard-Scape" },
			{7 , "Scanning-Artifacts" },
			{8 , "Cars" },
			{9 , "Utility-Pole" },
			{10 , "Insulator" },
			{11 , "Electrical-Wire" },
			{12 , "Cross-Bar" },
			{13 , "Stick" },
			{14 , "Fuse" },
			{15 , "Wire-clip" },
			{16 , "Linker-insulator" },
			{17 , "Persons" },
			{18 , "Traffic-Sign" },
			{19 , "Traffic-Light" }
	};

	static int FindindexByValue(const std::string& value)
	{
		std::map<size_t, std::string>::const_iterator it = SemanticMap.begin();
		for (; it != SemanticMap.end(); ++it)
		{
			if (it->second == value)
			{
				return static_cast<int>(it->first);
			}
		}

		return -1;
	}

	typedef std::map< std::string, std::vector<size_t> > ClusterMap;
}

