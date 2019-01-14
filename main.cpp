#include <iostream>
#include <future>

#include "recommender.h"
#include "crossValidation.h"

int main(int argc, char* argv[]) {
    UsersDataVector data;
    SongsVector songs;
    if (argc > 2) {
        readData(argv[1], data, songs);
    } else {
        readData("train_triplets.txt", data, songs);
    }

    std::cout << "Read data\n";

    auto copyData = UsersDataVector(data);

    auto rmse = std::async(
            std::launch::async,
            crossValidationRMSE,
            std::ref(data),
            predictFromNeighborsWithSongMark,
            50);
    auto ndcg_gini = std::async(
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
    auto ndcg_gini_val = ndcg_gini.get();
    std::cout << "nDCG score: " << ndcg_gini_val.first << '\n'
        << "Gini score: " << ndcg_gini_val.second << '\n';

    return 0;
}
