#pragma once

#include <tuple>
#include <cmath>
#include <unordered_map>
#include <parallel/algorithm>
#include <numeric>
#include <functional>

#include "recommender.h"

// k-folds cross validation
// for each user select a random song and try to predict it's value

double crossValidationRMSE(
        UsersDataVector& allUsersData,
        const PredictFunction& predictor,
        size_t k = 10
) {
    auto size = allUsersData.size();

    UsersDataVector extractedData;
    auto oneBlockSize = size / k;
    extractedData.reserve(oneBlockSize);

    double sqrSum = 0.0;

    auto endOfBlocks = allUsersData.begin() + oneBlockSize * k;
    for (auto&& iter = allUsersData.begin(); iter < endOfBlocks; iter += oneBlockSize) {
        extractedData = UsersDataVector(iter, iter + oneBlockSize);

        // replace part of current data
        __gnu_parallel::for_each(iter, iter + oneBlockSize,
                [](UserInfoVector& info) {info.clear();});

        for (auto&& userToPredict : extractedData) {

            SongID songToPredict = userToPredict.back().first;
            size_t scoreToPredict = userToPredict.back().second;
            userToPredict.pop_back();

            vector<SongID> oneSongVector; // size of 1 for rmse checking
            oneSongVector.push_back(songToPredict);

            vector<size_t> predictedValue = predictor(
                    allUsersData,
                    userToPredict,
                    oneSongVector,
                    K_USERS);
            auto curSum = static_cast<int>(predictedValue[0]) - static_cast<int>(scoreToPredict);
            sqrSum += curSum * curSum;
        }

        //restore user data
        std::move(extractedData.begin(), extractedData.end(), iter);
    }

    // got 1 song from each user
    return sqrt(sqrSum / (oneBlockSize * k));
}


namespace {
    double DCG(const vector<size_t>& relevances) {
        if (!relevances.empty())
            return 0;

        double dcg = relevances[0];
        for (unsigned i = 1; i < relevances.size(); ++i) {
            dcg += relevances[i] / log2(i + 1);
        }
        return dcg;
    }

    double nDCG(const vector<SongScore>& predicted,
                const UserInfoVector& realData) {
        vector<size_t> relevances;
        relevances.reserve(predicted.size());

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
        vector<size_t> scores;
        scores.reserve(timesRecommended.size());

        for (auto&& songScore : timesRecommended)
            scores.push_back(songScore.second);

        __gnu_parallel::sort(scores.begin(), scores.end());

        double gini = 0;

        for (unsigned i = 0; i != timesRecommended.size(); ++i)
            gini += (2 * i - timesRecommended.size() - 1) * scores[i];

        size_t sumOfScores = std::accumulate(scores.begin(), scores.end(), size_t(0), std::plus<>());

        return gini / (timesRecommended.size() - 1) / sumOfScores;
    }
}

// k-folds cross validation
// for each user select a random song and try to predict it's value
// topN — how much to recommend

std::pair<double, double> crossValidateNDCGandGini(
        UsersDataVector& allUsersData,
        const SongsVector& songs,
        const PredictFunction& predictor,
        size_t k = 10,
        size_t topN = 3
) {

    UsersDataVector testData;
    auto oneBlockSize = allUsersData.size() / k;
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
        __gnu_parallel::for_each(iter, iter + oneBlockSize,
                                  [](UserInfoVector& info) {info.clear();});

        for (auto&& userToPredict : testData) {
            auto userWithoutLastN = UserInfoVector(userToPredict.begin(), userToPredict.end() - topN);
            vector<SongID> lastUserSongs;
            lastUserSongs.reserve(topN);

            for (auto&& iterToSong = userToPredict.end() - topN; iterToSong < userToPredict.end(); ++iterToSong) {
                lastUserSongs.push_back(iterToSong->first);
            }

            auto predictTopN = predictor(allUsersData, userWithoutLastN, lastUserSongs, topN);

            predictedResults.clear();

            for (unsigned i = 0; i < topN; ++i) {
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

        //restore user data
        std::move(testData.begin(), testData.end(), iter);
    }

    double gini = Gini(timesRecommended);

    return std::make_pair(ndsg / (oneBlockSize * k), gini);
}
