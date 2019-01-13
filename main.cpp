#include <iostream>
#include <future>

#include <future>

#include "recommender.h"
#include "crossValidation.h"

int main(int argc, char* argv[]) {
    UsersDataVector data;
    SongsVector songs;
    if (argc > 2) {
        readData(argv[1], data, songs);
    } else {
        readData("../train_triplets.txt", data, songs);
    }

    std::cout << "Read data\n";

    auto copyData = UsersDataVector(data);

    auto rmse = std::async(
            std::launch::async,
            crossValidationRMSE,
            std::ref(data),
            predictFromNeighborsWithSongMark,
            50);
    auto ndsg_gini = std::async(
            std::launch::async,
            crossValidateNDCGandGini,
            std::ref(copyData),
            std::ref(songs),
            predictFromNeighborsWithSongMark,
            50, 5);

    std::cout << "Counting RMSE score\n";
    auto rmse_val = rmse.get();
    std::cout << "RMSE score: " << rmse_val << '\n';

    std::cout << "Counting nDCG and Gini scores\n";
    auto ndsg_gini_val = ndsg_gini.get();
    std::cout << "nDCG score: " << ndsg_gini_val.first << '\n'
        << "Gini score: " << ndsg_gini_val.second << '\n';

    return 0;
}
