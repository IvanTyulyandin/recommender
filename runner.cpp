#include <chrono>
#include <iostream>

#include "recommender.h"


int main(int argc, char* argv[]) {
    UsersDataVector data;
    auto start_time = std::chrono::steady_clock::now();
    if (argc > 2) {
        readData(argv[1], data);
    } else {
        readData("/home/ivan/CLionProjects/recommender/train_triplets.txt", data);
    }

    auto userToPredict = data.back();
    data.pop_back();

    SongID songToPredict = userToPredict.back().first;
    size_t valueToPredict = userToPredict.back().second;
    userToPredict.pop_back();

    std::cout << predictSongListening(data, userToPredict, songToPredict)
        << " expected " << valueToPredict << ' ';
    // printDataVectorWithNames(data);
    auto cur = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
    std::cout << "Time: " << duration.count() << " msec\n";
    return 0;
}