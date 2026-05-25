#pragma once

#include <cstdio>
#include <string>
#include <sys/types.h>
#include <vector>

class EmbedClient
{
public:
    EmbedClient() = default;
    ~EmbedClient();

    EmbedClient(const EmbedClient &) = delete;
    EmbedClient &operator=(const EmbedClient &) = delete;

    std::string start(const std::string &scriptPath,
                      const std::string &pythonBin = "python3");

    void stop();

    bool running() const noexcept { return pid_ > 0; }

    std::vector<float> embed(const std::string &text);

private:
    pid_t pid_ = -1;
    int wfd_ = -1;
    int rfd_ = -1;
    FILE *rfile_ = nullptr;
};
