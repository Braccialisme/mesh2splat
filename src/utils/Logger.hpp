///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
///////////////////////////////////////////////////////////////////////////////
//
// Logger
// ------
// Captures everything written to std::cout / std::cerr into an in-memory buffer
// so the app can show it in an ImGui window (there is no console window). Also
// forwards to the original stream buffer, so a console (when present) still
// works. Thread-safe (exportPly logs from a detached thread). The buffer is
// capped so a long session can't grow it without bound.
//
// Usage: call utils::initLog() once at startup; render utils::logText() in a UI
// window; utils::clearLog() to reset.

#pragma once

#include <streambuf>
#include <string>
#include <mutex>
#include <iostream>

namespace utils
{
    class LogBuf : public std::streambuf
    {
    public:
        std::string      text;
        std::streambuf*  orig = nullptr;
        std::mutex       mtx;
        static constexpr size_t kCap = 256 * 1024;   // keep last ~256 KB

    protected:
        int overflow(int c) override
        {
            if (c != EOF) {
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    text.push_back(static_cast<char>(c));
                    trim();
                }
                if (orig) orig->sputc(static_cast<char>(c));
            }
            return c;
        }
        std::streamsize xsputn(const char* s, std::streamsize n) override
        {
            {
                std::lock_guard<std::mutex> lk(mtx);
                text.append(s, static_cast<size_t>(n));
                trim();
            }
            if (orig) orig->sputn(s, n);
            return n;
        }

    private:
        void trim()   // caller holds mtx
        {
            if (text.size() > kCap) {
                size_t drop = text.size() - kCap;
                size_t nl = text.find('\n', drop);          // cut on a line boundary
                text.erase(0, nl == std::string::npos ? drop : nl + 1);
            }
        }
    };

    inline LogBuf& gLog()
    {
        static LogBuf buf;
        return buf;
    }

    inline void initLog()
    {
        gLog().orig = std::cout.rdbuf(&gLog());
        std::cerr.rdbuf(&gLog());
    }

    inline std::string logText()
    {
        std::lock_guard<std::mutex> lk(gLog().mtx);
        return gLog().text;
    }

    inline void clearLog()
    {
        std::lock_guard<std::mutex> lk(gLog().mtx);
        gLog().text.clear();
    }
}
