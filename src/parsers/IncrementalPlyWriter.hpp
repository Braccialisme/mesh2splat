///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////
//
// IncrementalPlyWriter
// --------------------
// Streams standard-format 3DGS gaussians (62 floats per vertex) to a binary
// little-endian PLY file, batch by batch, without ever holding the full set
// in memory. The vertex count is unknown until the last batch, so the header
// is written up front with a zero-padded 10-digit count field
// ("element vertex 0000000000") which finalize() patches in place with the
// real count. Leading zeros are a valid ASCII integer for PLY parsers.
//
// The per-vertex byte layout intentionally replicates
// parsers::writeBinaryPlyStandardFormat exactly:
//   x y z | nx ny nz | f_dc_0..2 (SH from color) | f_rest_0..44 (zeros)
//   | opacity (invSigmoid of alpha) | scale_0..2 (log(scale*mult)) | rot_0..3
//
// Usage:
//   IncrementalPlyWriter w;
//   if (!w.open(path, scaleMultiplier)) { ...error... }
//   w.appendBatch(batch0); w.appendBatch(batch1); ...
//   if (!w.finalize()) { ...error... }
// On cancellation call abort() to close and delete the partial file.

#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>

#include "utils/utils.hpp"

namespace parsers
{
    class IncrementalPlyWriter
    {
    public:
        IncrementalPlyWriter() = default;
        ~IncrementalPlyWriter() { if (m_file.is_open()) m_file.close(); }

        IncrementalPlyWriter(const IncrementalPlyWriter&) = delete;
        IncrementalPlyWriter& operator=(const IncrementalPlyWriter&) = delete;

        // Opens the file and writes the full header with a placeholder count.
        // Returns false if the file could not be opened or written.
        bool open(const std::string& filename, float scaleMultiplier)
        {
            m_filename        = filename;
            m_scaleMultiplier = scaleMultiplier;
            m_writtenCount    = 0;
            m_failed          = false;

            // Large stream buffer: must be set before open() to take effect.
            m_ioBuffer.resize(1 << 20); // 1 MB
            m_file.rdbuf()->pubsetbuf(m_ioBuffer.data(), static_cast<std::streamsize>(m_ioBuffer.size()));

            m_file.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!m_file.is_open()) { m_failed = true; return false; }

            m_file << "ply\n";
            m_file << "format binary_little_endian 1.0\n";
            m_file << "element vertex ";

            m_countFieldOffset = m_file.tellp();
            m_file << "0000000000\n"; // 10-digit placeholder, patched by finalize()

            m_file << "property float x\n";
            m_file << "property float y\n";
            m_file << "property float z\n";

            m_file << "property float nx\n";
            m_file << "property float ny\n";
            m_file << "property float nz\n";

            m_file << "property float f_dc_0\n";
            m_file << "property float f_dc_1\n";
            m_file << "property float f_dc_2\n";

            for (int i = 0; i <= 44; ++i) {
                m_file << "property float f_rest_" << i << "\n";
            }

            m_file << "property float opacity\n";

            m_file << "property float scale_0\n";
            m_file << "property float scale_1\n";
            m_file << "property float scale_2\n";

            m_file << "property float rot_0\n";
            m_file << "property float rot_1\n";
            m_file << "property float rot_2\n";
            m_file << "property float rot_3\n";

            m_file << "end_header\n";

            if (!m_file.good()) { m_failed = true; return false; }
            return true;
        }

        // Appends a batch of gaussians. Does not modify the input.
        // Returns false if a write error occurred (also latches m_failed).
        bool appendBatch(const std::vector<utils::GaussianDataSSBO>& gaussians)
        {
            if (m_failed || !m_file.is_open()) return false;

            float row[62];

            for (const auto& gaussian : gaussians)
            {
                // Mean
                row[0] = gaussian.position.x;
                row[1] = gaussian.position.y;
                row[2] = gaussian.position.z;

                // Normal
                row[3] = gaussian.normal.x;
                row[4] = gaussian.normal.y;
                row[5] = gaussian.normal.z;

                // RGB as SH degree-0 coefficients (same helper as legacy writer)
                glm::vec3 sh0 = utils::getShFromColor(gaussian.color);
                row[6] = sh0.r;
                row[7] = sh0.g;
                row[8] = sh0.b;

                // f_rest_0 .. f_rest_44 are zero (no higher-order SH)
                for (int i = 9; i <= 53; ++i) row[i] = 0.0f;

                // Opacity. invSigmoid(1.0f) is +inf in float precision (fully
                // opaque meshes hit this on every gaussian) and invSigmoid(0.0f)
                // is -inf. Clamp alpha so the logit stays finite (~ +/-13.8,
                // i.e. sigmoid 0.999999 / 0.000001 -- visually identical).
                {
                    const float eps   = 1e-6f;
                    float       alpha = gaussian.color.a;
                    alpha  = std::min(std::max(alpha, eps), 1.0f - eps);
                    row[54] = utils::invSigmoid(alpha);
                }

                // Scale: log(scale * multiplier), same math as legacy writer,
                // but computed locally so the caller's data is untouched.
                // Floor guards against log(0) = -inf on degenerate axes.
                {
                    const float minScale = 1e-12f; // log(1e-12) ~ -27.6, finite
                    row[55] = std::log(std::max(gaussian.scale.x * m_scaleMultiplier, minScale));
                    row[56] = std::log(std::max(gaussian.scale.y * m_scaleMultiplier, minScale));
                    row[57] = std::log(std::max(gaussian.scale.z * m_scaleMultiplier, minScale));
                }

                // Rotation quaternion
                row[58] = gaussian.rotation.x;
                row[59] = gaussian.rotation.y;
                row[60] = gaussian.rotation.z;
                row[61] = gaussian.rotation.w;

                m_file.write(reinterpret_cast<const char*>(row), sizeof(row));
            }

            m_writtenCount += static_cast<uint64_t>(gaussians.size());

            if (!m_file.good()) { m_failed = true; return false; }
            return true;
        }

        // Patches the real vertex count into the header and closes the file.
        // Returns false on any error (count too large, seek/write failure,
        // or a previously latched write failure).
        bool finalize()
        {
            if (m_failed || !m_file.is_open()) return false;

            if (m_writtenCount > 9999999999ULL) { m_failed = true; return false; }

            char countStr[11];
            std::snprintf(countStr, sizeof(countStr), "%010llu",
                          static_cast<unsigned long long>(m_writtenCount));

            m_file.seekp(m_countFieldOffset);
            m_file.write(countStr, 10);
            m_file.flush();

            bool ok = m_file.good();
            m_file.close();
            if (!ok) m_failed = true;
            return ok;
        }

        // Closes and deletes the partial file (for cancellation / error paths).
        void abort()
        {
            if (m_file.is_open()) m_file.close();
            std::remove(m_filename.c_str());
            m_failed = true;
        }

        uint64_t writtenCount() const { return m_writtenCount; }
        bool     failed()       const { return m_failed; }

    private:
        std::ofstream     m_file;
        std::vector<char> m_ioBuffer;
        std::string       m_filename;
        std::streampos    m_countFieldOffset{};
        float             m_scaleMultiplier = 1.0f;
        uint64_t          m_writtenCount    = 0;
        bool              m_failed          = false;
    };
}