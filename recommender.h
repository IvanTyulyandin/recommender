#pragma once

#include <vector>
#include <string>
#include <functional>

using std::vector;
using std::string;

// got this number from https://labrosa.ee.columbia.edu/millionsong/tasteprofile
constexpr size_t USERS = 264000;
constexpr size_t K_USERS = 20;
constexpr size_t TOP_K = 3;

typedef string SongID;
typedef std::pair<SongID, size_t> SongScore;

typedef vector<SongScore> UserInfoVector;
typedef vector<UserInfoVector> UsersDataVector;

typedef std::pair<double, size_t> CosValueIndex;
typedef vector<SongID> SongsVector;

using PredictFunction = std::function<vector<size_t>(
        const UsersDataVector &allUsersData,
        const UserInfoVector &userData,
        const vector<SongID> &songsID,
        size_t topK)>;

void readData(const string& inputFile, UsersDataVector& data, SongsVector& songs);

void printDataVectorWithNames(const UsersDataVector& data);

double cosBetweenTwoUsers(const UserInfoVector& fst, const UserInfoVector& snd);

vector<size_t> predictFromNeighborsWithSongMark(
        const UsersDataVector &allUsersData,
        const UserInfoVector &userData,
        const vector<SongID> &songsID,
        size_t topK = K_USERS // number of nearest users
);

vector<size_t> predictSongsFromNearestNeighbors(
        const UsersDataVector& allUsersData,
        const UserInfoVector& userData,
        const vector<SongID>& songsToPredict,
        size_t topK = K_USERS // number of nearest users
);
