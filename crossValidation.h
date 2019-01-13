#pragma once

#include <tuple>
#include <cmath>
#include <unordered_map>
#include <parallel/algorithm>

#include "recommender.h"

// k-folds cross validation
// for each user select a random song and try to predict it's value

double crossValidationRMSE(UsersDataVector& allUsersData, size_t k = 10) {
    auto size = allUsersData.size();

    UsersDataVector extractedData;
    auto oneBlockSize = size / k;
    extractedData.reserve(oneBlockSize);

    double sqrSum = 0.0;

    auto endOfBlocks = allUsersData.begin() + oneBlockSize * k;
    for (auto&& iter = allUsersData.begin(); iter < endOfBlocks; iter += oneBlockSize) {
        extractedData = UsersDataVector(iter, iter + oneBlockSize);

        // replace part of current data
        auto endOfReplacement = iter + oneBlockSize;
        __gnu_parallel::transform(iter, iter + oneBlockSize, iter,
                [](UserInfoVector& info) {info.clear(); return info;});

        for (auto&& userToPredict : extractedData) {

            SongID songToPredict = userToPredict.back().first;
            size_t scoreToPredict = userToPredict.back().second;
            userToPredict.pop_back();

            vector<SongID> oneSongVector; // size of 1 for rmse checking
            oneSongVector.push_back(songToPredict);

            vector<size_t> predictedValue = predictFromNeighborsWithSongMark(
                    allUsersData,
                    userToPredict,
                    oneSongVector);
            auto curSum = static_cast<int>(predictedValue[0]) - static_cast<int>(scoreToPredict);
            sqrSum += curSum * curSum;
        }
        std::cout << "RMSE is done\n";
        //restore user data
        std::move(extractedData.begin(), extractedData.end(), iter);
        std::cout << "move is done\n";
        break;
    }

    // got 1 song from each user
    return sqrt(sqrSum / (oneBlockSize));
}


namespace {
double DCG(const vector<size_t>& relevances) {
    size_t k = relevances.size();
    if (!k)
        return 0;

    double dcg = relevances[0];
    for (auto i = 1; i < k; ++i) {
        dcg += relevances[i] / log2(i + 1);
    }
    return dcg;
}

double nDCG(const vector<SongScore>& predicted,
            const UserInfoVector& realData) {
    auto N = predicted.size();
    vector<size_t> relevances;
    relevances.reserve(N);

    for (auto&& songScore : predicted) {
        auto it = std::find_if(realData.begin(), realData.end(),
                [&songScore](const SongScore& ss) { return songScore.first == ss.first;});
        if (it != realData.end())
            relevances.emplace_back(it->second);
        else
            relevances.emplace_back(0);
    }

    auto dcg = DCG(relevances);
    std::sort(relevances.begin(), relevances.end(), std::greater<>());
    auto idcg = DCG(relevances);
    //std::cout << dcg << ", " << idcg << '\n';
    return dcg / idcg;
}

double Gini(const std::unordered_map<SongID, size_t>& timesRecommended) {
    size_t size = timesRecommended.size();
    vector<size_t> scores;
    scores.reserve(size);

    for (auto&& songScore : timesRecommended)
        scores.push_back(songScore.second);

    __gnu_parallel::sort(scores.begin(), scores.end());

    double gini = 0;

    for (auto i = 0; i != size; ++i)
        gini += (2 * i - size - 1) * scores[i];

    return gini / (--size);
}
}

// k-folds cross validation
// for each user select a random song and try to predict it's value
// topN — how much to recommend

std::pair<double, double> crossValidateNDCGandGini(
        UsersDataVector& allUsersData,
        const SongsVector& songs,
        size_t k = 10,
        size_t topN = 3
) {
    auto size = allUsersData.size();

    UsersDataVector testData;
    auto oneBlockSize = size / k;
    testData.reserve(oneBlockSize);

    vector<SongScore> predictedResults;
    predictedResults.reserve(topN);

    std::unordered_map<SongID, size_t> timesRecommended;
    timesRecommended.reserve(songs.size());

    __gnu_parallel::for_each(songs.begin(), songs.end(),
            [&timesRecommended](const SongID& song) {
                timesRecommended[song] = 0;
    });

    double ndsg = 0;

    auto endOfBlocks = allUsersData.begin() + oneBlockSize * k;
    for (auto&& iter = allUsersData.begin(); iter < endOfBlocks; iter += oneBlockSize) {
        testData = UsersDataVector(iter, iter + oneBlockSize - 1);

        // replace part of current data
        auto endOfReplacement = iter + oneBlockSize - 1;
        __gnu_parallel::for_each(iter, iter + oneBlockSize,
                                  [](UserInfoVector& info) {info.clear();});

        for (auto&& userToPredict : testData) {
            auto userWithoutLastN = UserInfoVector(userToPredict.begin(), userToPredict.end() - topN);
            vector<SongID> lastUserSongs;
            lastUserSongs.reserve(topN);

            for (auto&& iterToSong = userToPredict.end() - topN; iterToSong < userToPredict.end(); ++iterToSong) {
                lastUserSongs.push_back(iterToSong->first);
            }

            auto predictTopN = predictSongsFromNearestNeighbors(allUsersData, userWithoutLastN, lastUserSongs);

            predictedResults.clear();

            for (auto i = 0; i < topN; ++i) {
                predictedResults.emplace_back(std::move(lastUserSongs[i]), predictTopN[i]);
            }

            std::sort(
                    predictedResults.begin(),
                    predictedResults.end(),
                    [](const SongScore& lhs, const SongScore& rhs) {
                        return lhs.second > rhs.second;
            });

            for (auto&& i : predictedResults) {
                timesRecommended[i.first] += 1;
            }

            ndsg += nDCG(predictedResults, userToPredict);
        }
        std::cout << "ndcg is done\n";
        //restore user data
        std::move(testData.begin(), testData.end(), iter);
        std::cout << "move is done\n";
        break;
    }
    double gini = Gini(timesRecommended);
    std::cout << "gini is done\n";

    return std::make_pair(ndsg / oneBlockSize, gini);
}

//
//double RMSEatLastUsers(UsersDataVector& allUsersData, size_t lastUsersNumber = 100) {
//
//    auto testDataBeginIter = allUsersData.end() - lastUsersNumber;
//    auto testDataEndIter = allUsersData.end();
//    auto lastUsersRealData = UsersDataVector(testDataBeginIter, testDataEndIter);
//
//    // replace part of current data
//    __gnu_parallel::for_each(testDataBeginIter, testDataEndIter,
//                             [](UserInfoVector& info) {info.clear();});
//
//    double sumOfErrors = 0.0;
//    for (auto&& userToPredict : lastUsersRealData) {
//        std::cout << "User done\n";
//
//        //take last song and try to predict it
//        auto realSongScore = userToPredict.back();
//        userToPredict.pop_back();
//
//        auto predicted = predictFromNeighborsWithSongMark(allUsersData, userToPredict, realSongScore.first);
//        sumOfErrors += pow(predicted - realSongScore.second, 2);
//
//        //restore data
//        userToPredict.push_back(realSongScore);
//    }
//
//    // restore data taken to test
//    std::move(lastUsersRealData.begin(), lastUsersRealData.end(), testDataBeginIter);
//
//    return sqrt(sumOfErrors / lastUsersNumber);
//}
//
//
//std::pair<double, double> NDCGandGiniatLastUsers(
//        UsersDataVector& allUsersData,
//        const SongsVector& songs,
//        size_t topN = TOP_K,
//        size_t lastUsersNumber = 100
//) {
//    vector<SongScore> predictedResults;
//    predictedResults.reserve(songs.size());
//
//    std::unordered_map<SongID, size_t> timesRecommended;
//    timesRecommended.reserve(songs.size());
//
//    __gnu_parallel::for_each(songs.begin(), songs.end(),
//                             [&timesRecommended](const SongID& song) {
//                                 timesRecommended[song] = 0;
//                             });
//
//    auto testDataBeginIter = allUsersData.end() - lastUsersNumber;
//    auto testDataEndIter = allUsersData.end();
//    auto lastUsersRealData = UsersDataVector(testDataBeginIter, testDataEndIter);
//
//    double ndsg = 0.0;
//
//    for (auto&& userToPredict : lastUsersRealData) {
//
//        auto userWithoutLastN = UserInfoVector(userToPredict.begin(), userToPredict.end() - topN);
//
//        predictedResults.clear();
//        for (auto&& song : songs) {
//            predictedResults.emplace_back(
//                    SongScore(song, predictFromNeighborsWithSongMark(allUsersData, userWithoutLastN, song)));
//            std::cout << "1 song prediction done\n";
//        }
//
//        __gnu_parallel::partial_sort(
//                predictedResults.begin(),
//                predictedResults.begin() + topN - 1,
//                predictedResults.end(),
//                [](const SongScore& lhs, const SongScore& rhs) {
//                    return lhs.second > rhs.second;
//                });
//
//        auto afterNth = predictedResults.begin() + topN;
//        for (auto&& i = predictedResults.begin(); i < afterNth; ++i) {
//            timesRecommended[i->first] += 1;
//        }
//
//        ndsg += nDCG(vector<SongScore>(predictedResults.begin(), afterNth), userToPredict);
//
//        std::cout << "User done\n";
//    }
//
//    std::move(lastUsersRealData.begin(), lastUsersRealData.end(), testDataBeginIter);
//
//    double gini = Gini(timesRecommended);
//
//    return std::make_pair(ndsg / lastUsersNumber, gini);
//}
