#pragma once

#include <string>
#include <vector>

// Split without modifying the input string; callers rely on empty fields being preserved.
inline std::vector<std::string> parse_string(const std::string& parse_target, const std::string& delimiter) {
    std::vector<std::string> parsed_content;
    // An empty delimiter would make find() succeed forever, so keep the whole string as one token.
    if (delimiter.empty()) {
        parsed_content.push_back(parse_target);
        return parsed_content;
    }

    // Move a read cursor through the string instead of repeatedly erasing from the front.
    std::size_t start = 0;
    std::size_t pos = parse_target.find(delimiter, start);
    while (pos != std::string::npos) {
        parsed_content.push_back(parse_target.substr(start, pos - start));
        start = pos + delimiter.length();
        pos = parse_target.find(delimiter, start);
    }
    parsed_content.push_back(parse_target.substr(start));

    return parsed_content;
}
