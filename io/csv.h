#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace galactic_nuclear_fountain::csv {

using Row=std::unordered_map<std::string,std::string>;

struct Table {
    std::vector<std::string> header;
    std::vector<Row> rows;
};

Table read(const std::filesystem::path& path);
const std::string& value(const Row& row,const std::string& name);
double number(const Row& row,const std::string& name);
std::string escape(const std::string& value);

}
