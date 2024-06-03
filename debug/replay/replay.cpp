#include "board.hpp"
#include "uci.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace enyo;

int extractDepth(const std::string& line) {
    size_t depthPos = line.find("depth");
    if (depthPos != std::string::npos) {
        size_t startPos = depthPos + 6; // "depth " is 6 characters long
        size_t spacePos = line.find(' ', startPos);
        if (spacePos != std::string::npos) {
            std::string depthStr = line.substr(startPos, spacePos - startPos);
            int depth = std::stoi(depthStr);
            return depth;
        }
    }
    return -1;
}

std::string extractMove(const std::string& line) {
    size_t bestMovePos = line.find("bestmove ");
    if (bestMovePos != std::string::npos) {
        size_t startPos = bestMovePos + 9; // "bestmove " is 9 characters long
        std::string move = line.substr(startPos);
        return move;
    }
    return "";
}

int main(int argc, char* argv[])
{
    Board b{"startpos"};
    Uci uci(b);
    NNUE::Init("");

    std::string logfile = "/tmp/enyo.log";
    if (argc > 1) {
        logfile = argv[1];
    }

    std::ifstream file(logfile);
    if (!file.is_open()) {
        std::cout << "Failed to open the logfile: " << logfile << std::endl;
        return 1;
    }

    std::vector<std::string> commands;
    std::vector<std::string> bestmoves;
    std::string line;
    std::streampos lastUciNewGamePos = -1;

    while (std::getline(file, line)) {
        if (line == "ucinewgame") {
            lastUciNewGamePos = file.tellg();
        }
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    if (lastUciNewGamePos != -1) {
        file.seekg(lastUciNewGamePos);
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::string command = line.substr(0, line.find(' '));
        if (command == "position" || command == "go" || command == "ucinewgame") {
            if (command != "go") {
                commands.push_back(line);
            } else {
                std::string depth_str;
                int depth = 0;
                while (std::getline(file, depth_str)) {
                    if (depth_str.find("bestmove ") != std::string::npos) {
                        bestmoves.push_back(extractMove(depth_str));
                        break;
                    }
                    if (depth_str.find("info depth ") != std::string::npos) {
                        depth = extractDepth(depth_str);
                        //fmt::print("[{}] info depth: {}\n", depth, depth_str);
                    }
                }
                //fmt::print("MAXDEPTH: {}\n", depth);
                commands.push_back(fmt::format("go depth {}", depth));
            }
        }
    }

    file.close();
    int bestmoveIndex = 0;

    //std::cout << "Supported commands found after the last 'ucinewgame':" << std::endl;
    for (const auto& cmd : commands) {

        std::cout << "uci:> " << cmd << std::endl;
        uci(cmd);

        if (cmd.find("go depth") != std::string::npos) {
            fmt::print("bestmove {} (EXPECTED)\n", bestmoves[bestmoveIndex++]);
        }
    }
    uci("d");
    uci("pgn");

    return 0;
}
