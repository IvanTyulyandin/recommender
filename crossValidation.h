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
    size_t trainDataSize = 0;

    auto endOfBlocks = allUsersData.begin() + oneBlockSize * k;
    for (auto&& iter = allUsersData.begin(); iter < endOfBlocks; iter += oneBlockSize) {
        extractedData = UsersDataVector(iter, iter + oneBlockSize);

        // replace part of current data
        auto endOfReplacement = iter + oneBlockSize;
        __gnu_parallel::transform(iter, iter + oneBlockSize, iter,
                [](UserInfoVector& info) {info.clear(); return info;});

        for (auto&& userToPredict : extractedData) {

            for (auto&& iterToDataToPredict = userToPredict.begin(); iterToDataToPredict != userToPredict.end();) {
                SongID songToPredict = iterToDataToPredict->first;
                size_t scoreToPredict = iterToDataToPredict->second;

                iterToDataToPredict = userToPredict.erase(iterToDataToPredict);

                size_t predictedValue = predictSongListening(allUsersData, userToPredict, songToPredict);
                sqrSum += pow((predictedValue - scoreToPredict), 2);

                // restore deleted data
                // iterToDataToPredict now points to restored data
                iterToDataToPredict = userToPredict.insert(iterToDataToPredict, SongScore(songToPredict, scoreToPredict));
                ++iterToDataToPredict;
            }

            trainDataSize += userToPredict.size();
        }

        //restore user data
        std::move(extractedData.begin(), extractedData.end(), iter);
    }

    return sqrt(sqrSum / trainDataSize);
}


namespace {
double DCG(const vector<size_t>& relevances) {
    size_t k = relevances.size();
    double dcg = 0;
    for (auto i = 0; i != k; ++i)
        dcg += relevances[i] / log2(i + 1);
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
    predictedResults.reserve(songs.size());

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

            predictedResults.clear();
            for (auto&& song : songs) {
                predictedResults.emplace_back(
                        SongScore(song, predictSongListening(allUsersData, userWithoutLastN, song)));
            }

            __gnu_parallel::partial_sort(
                    predictedResults.begin(),
                    predictedResults.begin() + topN - 1,
                    predictedResults.end(),
                    [](const SongScore& lhs, const SongScore& rhs) {
                        return lhs.second > rhs.second;
            });

            auto afterNth = predictedResults.begin() + topN;
            for (auto&& i = predictedResults.begin(); i < afterNth; ++i) {
                timesRecommended[i->first] += 1;
            }

            ndsg += nDCG(vector<SongScore>(predictedResults.begin(), afterNth), userToPredict);
        }

        //restore user data
        std::move(testData.begin(), testData.end(), iter);
    }
    double gini = Gini(timesRecommended);

    return std::make_pair(ndsg / allUsersData.size(), gini);
}
