///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////
//
// NativeFileDialog
// ----------------
// Thin wrapper around the native Windows file/folder picker (IFileOpenDialog,
// the real Explorer dialog: pinned folders, search, previews -- everything).
// Replaces the in-app ImGuiFileDialog for a much better user experience.
//
// All functions are blocking (the app freezes while the dialog is open, same
// as any modal dialog) and return std::nullopt if the user cancels.
// Returned paths are UTF-8 encoded std::strings.
//
// On non-Windows platforms these compile to stubs that always return nullopt.

#pragma once

#include <string>
#include <optional>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX            // keep windows.h from clobbering std::min/std::max
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shobjidl.h>       // IFileOpenDialog
#endif

namespace nativeDialog
{
#ifdef _WIN32

    // UTF-16 -> UTF-8 conversion for the returned path.
    inline std::string wideToUtf8(const wchar_t* wide)
    {
        if (!wide) return {};
        int len = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
        if (len <= 1) return {};
        std::string out(static_cast<size_t>(len - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), len, nullptr, nullptr);
        return out;
    }

    // Shared implementation for "open file" and "pick folder".
    // filters may be empty (e.g. for folder picking).
    inline std::optional<std::string> runDialog(
        bool pickFolders,
        const wchar_t* title,
        const std::vector<COMDLG_FILTERSPEC>& filters)
    {
        std::optional<std::string> result;

        // COM may or may not already be initialized on this thread.
        // S_OK / S_FALSE both need a balancing CoUninitialize;
        // RPC_E_CHANGED_MODE means someone initialized with another model --
        // we can still use the dialog, but must not uninitialize.
        HRESULT hrInit    = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        bool    balanceIt = SUCCEEDED(hrInit);

        IFileOpenDialog* dialog = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                       IID_IFileOpenDialog, reinterpret_cast<void**>(&dialog))))
        {
            DWORD options = 0;
            dialog->GetOptions(&options);
            options |= FOS_FORCEFILESYSTEM;             // real paths only
            if (pickFolders) options |= FOS_PICKFOLDERS;
            dialog->SetOptions(options);

            if (title) dialog->SetTitle(title);
            if (!filters.empty())
                dialog->SetFileTypes(static_cast<UINT>(filters.size()), filters.data());

            // GetActiveWindow(): the user just clicked a button in our window,
            // so it is the active one -- this parents the dialog correctly
            // without needing GLFW native-handle plumbing.
            if (SUCCEEDED(dialog->Show(GetActiveWindow())))
            {
                IShellItem* item = nullptr;
                if (SUCCEEDED(dialog->GetResult(&item)))
                {
                    PWSTR path = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)))
                    {
                        result = wideToUtf8(path);
                        CoTaskMemFree(path);
                    }
                    item->Release();
                }
            }
            dialog->Release();
        }

        if (balanceIt) CoUninitialize();
        return result;
    }

    // "Select file to load" -- .glb / .ply models.
    inline std::optional<std::string> openModelFile()
    {
        std::vector<COMDLG_FILTERSPEC> filters = {
            { L"3D models (*.glb, *.ply)", L"*.glb;*.ply" },
            { L"glTF binary (*.glb)",      L"*.glb"       },
            { L"Gaussian splat (*.ply)",   L"*.ply"       },
            { L"All files (*.*)",          L"*.*"         },
        };
        return runDialog(false, L"Choose file to load", filters);
    }

    // Folder picker.
    inline std::optional<std::string> pickFolder(const wchar_t* title)
    {
        return runDialog(true, title, {});
    }

#else // non-Windows stubs

    inline std::optional<std::string> openModelFile()                 { return std::nullopt; }
    inline std::optional<std::string> pickFolder(const wchar_t*)      { return std::nullopt; }

#endif
}