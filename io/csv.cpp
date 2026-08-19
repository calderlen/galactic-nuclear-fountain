#include "csv.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace galacticwind::csv {
namespace {

std::vector<std::string> parse_line(const std::string& line){
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false;
    for (std::size_t index=0; index<line.size(); ++index) {
        const char character=line[index];
        if (character=='"') {
            if (quoted && index+1<line.size() && line[index+1]=='"') {
                field.push_back('"');
                ++index;
            } else {
                quoted=!quoted;
            }
        } else if (character==',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else if (character!='\r') {
            field.push_back(character);
        }
    }
    fields.push_back(field);
    return fields;
}

}

Table read(const std::filesystem::path& path){
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Could not read "+path.string());
    }
    Table table;
    std::string line;
    std::getline(stream,line);
    table.header=parse_line(line);
    while (std::getline(stream,line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields=parse_line(line);
        Row row;
        for (std::size_t index=0; index<table.header.size(); ++index) {
            row[table.header[index]]=index<fields.size() ? fields[index] : "";
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

const std::string& value(const Row& row,const std::string& name){
    return row.at(name);
}

double number(const Row& row,const std::string& name){
    return std::stod(value(row,name));
}

std::string escape(const std::string& value){
    if (value.find_first_of(",\"\n\r")==std::string::npos) {
        return value;
    }
    std::string escaped="\"";
    for (const char character: value) {
        escaped+=character=='\"' ? "\"\"" : std::string(1,character);
    }
    return escaped+"\"";
}

}
