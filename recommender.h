#pragma once

#include <vector>
#include <string>

using std::vector;
using std::string;

// got this number from https://labrosa.ee.columbia.edu/millionsong/tasteprofile
constexpr size_t USERS = 1'019'318;
constexpr size_t K_USERS = 20;
constexpr size_t TOP_K = 3;

typedef string SongID;
typedef std::pair<SongID, size_t> SongScore;

typedef vector<SongScore> UserInfoVector;
typedef vector<UserInfoVector> UsersDataVector;

typedef std::pair<double, size_t> CosValueIndex;
typedef vector<SongID> SongsVector;


void readData(const string& inputFile, UsersDataVector& data, SongsVector& songs);

void printDataVectorWithNames(const UsersDataVector& data);

double cosBetweenTwoUsers(const UserInfoVector& fst, const UserInfoVector& snd);

size_t predictFromNeighborsWithSongMark(
        const UsersDataVector &allUsersData,
        const UserInfoVector &userData,
        const SongID &songID,
        size_t topK = K_USERS
);

vector<size_t> predictSongsFromNearestNeighbors(
        const UsersDataVector& allUsersData,
        const UserInfoVector& userData,
        const vector<SongID>& songsToPredict,
        size_t topK = K_USERS
);