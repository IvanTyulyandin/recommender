// #include <chrono>
#include <iostream>

#include <future>

#include "recommender.h"
#include "crossValidation.h"

int main(int argc, char* argv[]) {
    UsersDataVector data;
    SongsVector songs;
//    auto start_time = std::chrono::steady_clock::now();
    if (argc > 2) {
        readData(argv[1], data, songs);
    } else {
        readData("train_triplets.txt", data, songs);
    }

    std::cout << "Read data\n";

    auto rmse = std::async(std::launch::async, crossValidationRMSE, std::ref(data), static_cast<size_t>(10));
    std::cout << "RMSE: " << rmse.get() << "\n";
    auto ndsg_gini = crossValidateNDCGandGini(data, songs);
    std::cout << "NDCG: " << ndsg_gini.first << "\nGINI: " << ndsg_gini.second << '\n';
    // printDataVectorWithNames(data);
//    auto cur = std::chrono::steady_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
//    std::cout << "Time: " << duration.count() << " msec\n";
    return 0;
}
