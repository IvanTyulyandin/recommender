#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <parallel/algorithm>
#include <unordered_set>
#include <mutex>

#include "recommender.h"


void readData(const string& inputFile, UsersDataVector& data, SongsVector& songs) {

    std::ifstream fileStream(inputFile, std::fstream::in);
    if (fileStream.fail()) {
        std::cout << "Can't open file " << inputFile << '\n';
        return;
    }

    string userID;
    string songID;
    size_t score;
    string prevUserID = userID;

    std::unordered_set<SongID> uniqueSongs;
    data = UsersDataVector();
    data.reserve(USERS);

    // suppose file structure is "userID \t songID \t score \n"
    // \t is whitespace symbol, so stream will skip \t
    while (fileStream >> userID >> songID >> score) {
        uniqueSongs.insert(songID);
        if (prevUserID != userID) {
            prevUserID = userID;
            data.push_back(UserInfoVector());
            data.back().reserve(10);
            // 10 values SongScore at least for user, took this magic number from dataset docs
        }
        data.back().push_back(SongScore(std::move(songID), score));
    }

    data.shrink_to_fit();
    songs = SongsVector(uniqueSongs.begin(), uniqueSongs.end());
    // don't need to sort, dataset is sorted already
    fileStream.close();
}


void printDataVectorWithNames(const UsersDataVector& data) {
    using std::cout;

    for (auto&& i : data) {
        for (auto&& songAndScore : i) {
            cout << songAndScore.first << ' ' << songAndScore.second << '\n';
        }
        cout << "\n\n";
    }
}


double cosBetweenTwoUsers(const UserInfoVector& fst, const UserInfoVector& snd) {

    // if no any info about a user set cosine to 0
    if (fst.empty() || snd.empty()) {
        return 0;
    }

    size_t fstSumMarksSqr = 0;
    size_t sndSumMarksSqr = 0;
    size_t sumCommonMarksMult = 0;
    auto fstIter = fst.begin();
    auto sndIter = snd.begin();

    bool fstMarkWasAdded = false;
    bool sndMarkWasAdded = false;

    // use fact that UserInfoVector are sorted by name of song

    while ((fstIter != fst.end()) && sndIter != snd.end()) {
        const auto& fstSong = fstIter->first;
        const auto& sndSong = sndIter->first;

        if (! fstMarkWasAdded) {
            fstSumMarksSqr += fstIter->second * fstIter->second;
            fstMarkWasAdded = true;
        }
        if (! sndMarkWasAdded) {
            sndSumMarksSqr += sndIter->second * sndIter->second;
            sndMarkWasAdded = true;
        }

        if (fstSong == sndSong) {
            sumCommonMarksMult += fstIter->second * sndIter->second;
            ++fstIter;
            ++sndIter;
            fstMarkWasAdded = false;
            sndMarkWasAdded = false;
        } else if (fstSong < sndSong) {
            ++fstIter;
            fstMarkWasAdded = false;
        } else {
            ++sndIter;
            sndMarkWasAdded = false;
        }
    }

    return sumCommonMarksMult / sqrt(fstSumMarksSqr * sndSumMarksSqr);
}


vector<size_t> predictFromNeighborsWithSongMark(
        const UsersDataVector &allUsersData,
        const UserInfoVector &userData,
        const vector<SongID> &songsID,
        size_t topK // topK has default value == K_USERS
) {
    auto songID = songsID[0];
    size_t curIndex = 0;
    vector<CosValueIndex> cosResult;
    cosResult.reserve(allUsersData.size());

    std::mutex cosResultMutex;
    __gnu_parallel::for_each(allUsersData.begin(), allUsersData.end(),
            [&cosResult, &cosResultMutex, &curIndex, &userData]
            (const UserInfoVector& user) {
                auto cosValue = cosBetweenTwoUsers(userData, user);
                cosResultMutex.lock();
                cosResult.emplace_back(CosValueIndex(cosValue, curIndex));
                ++curIndex;
                cosResultMutex.unlock();
    });

    // having song > not having song, else compare by cos value
    auto cosComparerWithSong = [&allUsersData, &songID](const CosValueIndex& lhs, const CosValueIndex& rhs) -> bool {
        const UserInfoVector& lhsInfo = allUsersData[lhs.second];
        const UserInfoVector& rhsInfo = allUsersData[rhs.second];

        auto sameSongName = [&songID](const SongScore& it) -> bool {
            return it.first == songID;
        };

        bool isExistsInLhs = std::find_if(lhsInfo.begin(), lhsInfo.end(), sameSongName) != lhsInfo.end();

        bool isExistsInRhs = std::find_if(rhsInfo.begin(), rhsInfo.end(), sameSongName) != rhsInfo.end();

        if (isExistsInLhs ^ isExistsInRhs) {
            // if not same isExistsInLhs show if lhs precedes rhs
            // in straight-forward comparison need one more "if" :)
            return isExistsInLhs;
        } else return lhs.first > rhs.first;
    };

    // waiting for c++ 17 in 2019
    __gnu_parallel::nth_element(cosResult.begin(), cosResult.begin() + topK - 1, cosResult.end(), cosComparerWithSong);

    size_t sumOfMarks = 0;
    auto elemAfterKth = cosResult.begin() + topK;
    for (auto&& i = cosResult.begin(); i < elemAfterKth; ++i) {
        const auto& user = allUsersData[i->second];

        // beware of similar users without mark of sondID
        auto iterSongScore = std::find_if(user.begin(), user.end(), [&songID](const SongScore& it) -> bool {
            return it.first == songID;
        });

        if (iterSongScore != user.end()) {
            sumOfMarks += iterSongScore->second;
        }
    }

    vector<size_t> mark;
    mark.push_back(sumOfMarks / topK);
    return mark;

}

namespace {
    // return top k nearestNeighbors as vector of indexes
    vector<size_t> nearestNeighbors(
            const UsersDataVector& allUsersData,
            const UserInfoVector& userData,
            size_t k = TOP_K
    ) {
        size_t curIndex = 0;
        vector<CosValueIndex> cosResult;
        cosResult.reserve(allUsersData.size());
        for (auto &&user : allUsersData) {
            cosResult.emplace_back(CosValueIndex(cosBetweenTwoUsers(userData, user), curIndex));
            ++curIndex;
        }

        // for descending sort
        auto cosComparer = [](const CosValueIndex &lhs, const CosValueIndex &rhs) {
            return lhs.first > rhs.first;
        };

        __gnu_parallel::partial_sort(cosResult.begin(), cosResult.begin() + k - 1, cosResult.end(), cosComparer);

        vector<size_t> result;
        result.reserve(k);

        auto afterKth = cosResult.begin() + k;
        for (auto &&iter = cosResult.begin(); iter < afterKth; ++iter) {
            result.push_back(iter->second);
        }

        return result;
    }


    size_t predictOneSongFromNeighbors(
            const UsersDataVector& allUsersData,
            const vector<size_t>& neighbors,
            const SongID& songToPredict
    ) {
        size_t sumOfNeighborsMarks = 0;

        for (auto&& index : neighbors) {
            const auto& curUser = allUsersData[index];
            auto songScoreIter = std::find_if(curUser.begin(), curUser.end(),
                    [&songToPredict](const SongScore& ss) {
                        return ss.first == songToPredict;
            });

            if (songScoreIter != curUser.end())
                sumOfNeighborsMarks += songScoreIter->second;
            // else sumOfNeighborsMarks += 0
        }

        return sumOfNeighborsMarks / neighbors.size();
    }
}


vector<size_t> predictSongsFromNearestNeighbors(
        const UsersDataVector& allUsersData,
        const UserInfoVector& userData,
        const vector<SongID>& songsToPredict,
        size_t topK // has default value K_USERS
) {
    auto topNeighbors = nearestNeighbors(allUsersData, userData, topK);
    vector<size_t> result;
    result.reserve(songsToPredict.size());

    for (auto&& song : songsToPredict) {
        result.push_back(predictOneSongFromNeighbors(allUsersData, topNeighbors, song));
    }

    return result;
}
