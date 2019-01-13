// #include <chrono>
#include <iostream>
#include <future>

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
        readData("../train_triplets.txt", data, songs);
    }

    std::cout << "Read data\n";

    auto copyData = UsersDataVector(data);

    //auto rmse = std::async(std::launch::async, crossValidationRMSE, std::ref(data), size_t(50));
    auto dsng_gini = std::async(std::launch::async, crossValidateNDCGandGini, std::ref(copyData), std::ref(songs),
                                size_t(50), size_t(5));

    //auto rmse_val = rmse.get();
    auto ndsg_gini = dsng_gini.get();
    //std::cout << "RMSE: " << rmse_val << "\n";
    std::cout << "NDCG: " << ndsg_gini.first << "\nGINI: " << ndsg_gini.second << '\n';
    // printDataVectorWithNames(data);
//    auto cur = std::chrono::steady_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
//    std::cout << "Time: " << duration.count() << " msec\n";
    return 0;
}
