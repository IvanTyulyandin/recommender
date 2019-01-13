#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <parallel/algorithm>
#include <unordered_set>

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

    auto size = data.size();
    for (auto i = 0; i < size; ++i) {
        for (auto&& songAndScore : data[i]) {
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


size_t predictSongListening(
        const UsersDataVector& allUsersData,
        const UserInfoVector& userData,
        const SongID& songID,
        size_t topK // topK has default value == K_USERS
) {
    size_t curIndex = 0;
    vector<CosValueIndex> cosResult;
    cosResult.reserve(allUsersData.size());
    for (auto&& user : allUsersData) {
        cosResult.emplace_back(CosValueIndex(cosBetweenTwoUsers(userData, user), curIndex));
        ++curIndex;
    }

    // having song > not having song, else compare by cos value
    auto cosComparer = [&allUsersData, &songID](const CosValueIndex& lhs, const CosValueIndex& rhs) -> bool {
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
    __gnu_parallel::nth_element(cosResult.begin(), cosResult.begin() + topK - 1, cosResult.end(), cosComparer);

    size_t sumOfMarks = 0;
    auto elemAfterKth = cosResult.begin() + topK;
    for (auto&& i = cosResult.begin(); i < elemAfterKth; ++i) {
        auto user = allUsersData[i->second];

        // beware of similar users without mark of sondID
        auto iterSongScore = std::find_if(user.begin(), user.end(), [&songID](const SongScore& it) -> bool {
            return it.first == songID;
        });

        if (iterSongScore != user.end()) {
            sumOfMarks += iterSongScore->second;
        }
    }

    return sumOfMarks / topK;

}