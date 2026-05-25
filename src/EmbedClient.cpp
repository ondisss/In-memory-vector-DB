#include "EmbedClient.hpp"

#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>

EmbedClient::~EmbedClient() { stop(); }

std::string EmbedClient::start(const std::string &scriptPath,
                               const std::string &pythonBin)
{
    if (running())
        return "";

    int pipe_in[2], pipe_out[2];
    if (pipe(pipe_in) < 0)
        return "pipe() selhal (vstup)";
    if (pipe(pipe_out) < 0)
    {
        close(pipe_in[0]);
        close(pipe_in[1]);
        return "pipe() selhal (výstup)";
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        close(pipe_in[0]);
        close(pipe_in[1]);
        close(pipe_out[0]);
        close(pipe_out[1]);
        return "fork() selhal";
    }

    if (pid == 0)
    {
        dup2(pipe_in[0], STDIN_FILENO);
        dup2(pipe_out[1], STDOUT_FILENO);
        int log = open("embed_server.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log >= 0)
            dup2(log, STDERR_FILENO);
        close(pipe_in[1]);
        close(pipe_out[0]);
        execlp(pythonBin.c_str(), pythonBin.c_str(), scriptPath.c_str(), nullptr);
        _exit(127);
    }

    close(pipe_in[0]);
    close(pipe_out[1]);

    pid_ = pid;
    wfd_ = pipe_in[1];
    rfd_ = pipe_out[0];

    {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(rfd_, &fds);
        struct timeval tv{30, 0};
        int sel = select(rfd_ + 1, &fds, nullptr, nullptr, &tv);
        if (sel <= 0)
        {
            stop();
            return sel == 0
                       ? "Timeout – Python server se nespustil za 30 s (viz embed_server.log)"
                       : "select() selhal";
        }
    }

    rfile_ = fdopen(rfd_, "r");
    if (!rfile_)
    {
        close(rfd_);
        rfd_ = -1;
        stop();
        return "fdopen() selhal";
    }
    rfd_ = -1;

    char buf[256];
    if (!fgets(buf, sizeof(buf), rfile_) || strncmp(buf, "READY", 5) != 0)
    {
        stop();
        return "Python server neodpověděl správně (viz embed_server.log)";
    }

    return "";
}

void EmbedClient::stop()
{
    if (wfd_ >= 0)
    {
        close(wfd_);
        wfd_ = -1;
    }
    if (rfile_)
    {
        fclose(rfile_);
        rfile_ = nullptr;
    }
    else if (rfd_ >= 0)
    {
        close(rfd_);
        rfd_ = -1;
    }
    if (pid_ > 0)
    {
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }
}

std::vector<float> EmbedClient::embed(const std::string &text)
{
    if (!running())
        return {};

    std::string line = text + "\n";
    if (write(wfd_, line.c_str(), line.size()) < 0)
        return {};

    char buf[65536];
    if (!fgets(buf, sizeof(buf), rfile_))
        return {};

    std::vector<float> result;
    std::istringstream iss(buf);
    float x;
    while (iss >> x)
        result.push_back(x);
    return result;
}
